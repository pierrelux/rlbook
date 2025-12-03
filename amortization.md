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

However, this framework breaks down when the action space becomes large or continuous. Computing Bellman targets requires evaluating $\max_{a' \in \mathcal{A}} q(s',a';\boldsymbol{\theta})$ for each next state $s'$. When actions are continuous ($\mathcal{A} \subset \mathbb{R}^m$), this maximization requires solving a nonlinear program at every target computation. For a replay buffer with millions of transitions, this becomes computationally prohibitive.

This chapter addresses the continuous action problem while maintaining the FQI framework. We develop several approaches, unified by a common theme: **amortization**. Rather than solving the optimization problem $\max_a q(s,a;\boldsymbol{\theta})$ repeatedly at inference time, we invest computational effort during training to learn a mapping that directly produces good actions. This trades training-time cost for inference-time speed.

The strategies we examine are:

1. **Explicit optimization** (Section 2): Solve the maximization numerically for a subset of states, accepting the computational cost for exact solutions.

2. **Policy network amortization** (Sections 3-5): Learn a deterministic or stochastic policy network $\pi_{\boldsymbol{w}}$ that approximates $\arg\max_a q(s,a;\boldsymbol{\theta})$ or the optimal stochastic policy, enabling fast action selection via a single forward pass. This includes both hard-max methods (DDPG, TD3) and soft-max methods (SAC, PCL).

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

While this strategy allows us to handle very large or even infinite (continuous) state spaces, it still requires maximizing over actions ($\max_{a \in \mathcal{A}}$) during the dataset creation when computing the operator $\Bellman$ for each basepoint. This maximization becomes computationally expensive for large action spaces. We can address this by adding another level of optimization: for each sample added to our regression dataset, we employ numerical optimization methods to find actions that maximize the Bellman operator for the given state.

```{prf:algorithm} Fitted Q-Iteration with Explicit Optimization
:label: fitted-q-iteration-explicit

**Input:** MDP $(S, \mathcal{A}, P, R, \gamma)$, base points $\mathcal{B}$, function approximator $q(s,a; \boldsymbol{\theta})$, maximum iterations $N$, tolerance $\varepsilon > 0$

**Output:** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D} \leftarrow \emptyset$ $\quad$ // Regression dataset
    2. **for** each $(s,a,r,s') \in \mathcal{B}$ **do** $\quad$ // Monte Carlo with one sample
        1. $y_{s,a} \leftarrow r + \gamma \texttt{maximize}(q(s', \cdot; \boldsymbol{\theta}_n))$
        2. $\mathcal{D} \leftarrow \mathcal{D} \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D})$
    4. $\delta \leftarrow \frac{1}{|\mathcal{D}||\mathcal{A}|}\sum_{(s,a) \in \mathcal{D} \times \mathcal{A}} (q(s,a; \boldsymbol{\theta}_{n+1}) - q(s,a; \boldsymbol{\theta}_n))^2$
    5. $n \leftarrow n + 1$
4. **until** $\delta < \varepsilon$ or $n \geq N$
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

**Input:** MDP $(S, \mathcal{A}, P, R, \gamma)$, base points $\mathcal{B}$, subset for exact optimization $\mathcal{B}_{\text{opt}} \subset \mathcal{B}$, Q-function approximator $q(s,a; \boldsymbol{\theta})$, policy approximator $\pi_{\boldsymbol{w}}$, maximum iterations $N$, tolerance $\varepsilon > 0$

**Output:** Parameters $\boldsymbol{\theta}$ for Q-function, $\boldsymbol{w}$ for policy

1. Initialize $\boldsymbol{\theta}_0$, $\boldsymbol{w}_0$
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D}_q \leftarrow \emptyset$ $\quad$ // Q-function regression dataset
    2. $\mathcal{D}_\pi \leftarrow \emptyset$ $\quad$ // Policy regression dataset
    3. **for** each $(s,a,r,s') \in \mathcal{B}$ **do**
        1. **// Determine next state's action via exact optimization or approximation**
        2. **if** $s' \in \mathcal{B}_{\text{opt}}$ **then**
            1. $a^*_{s'} \leftarrow \texttt{maximize}(q(s', \cdot; \boldsymbol{\theta}_n))$
            2. $\mathcal{D}_\pi \leftarrow \mathcal{D}_\pi \cup \{(s', a^*_{s'})\}$
        3. **else**
            1. $a^*_{s'} \leftarrow \pi_{\boldsymbol{w}_n}(s')$
        4. **// Compute Q-function target using chosen action**
        5. $y_{s,a} \leftarrow r + \gamma q(s', a^*_{s'}; \boldsymbol{\theta}_n)$
        6. $\mathcal{D}_q \leftarrow \mathcal{D}_q \cup \{((s,a), y_{s,a})\}$
    4. **// Update both function approximators**
    5. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}_q)$
    6. $\boldsymbol{w}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}_\pi)$
    7. **// Compute convergence criteria**
    8. $\delta_q \leftarrow \frac{1}{|\mathcal{D}_q|}\sum_{(s,a) \in \mathcal{D}_q} (q(s,a; \boldsymbol{\theta}_{n+1}) - q(s,a; \boldsymbol{\theta}_n))^2$
    9. $\delta_\pi \leftarrow \frac{1}{|\mathcal{D}_\pi|}\sum_{(s,a^*) \in \mathcal{D}_\pi} \|a^* - \pi_{\boldsymbol{w}_{n+1}}(s)\|^2$
    10. $n \leftarrow n + 1$
4. **until** $\max(\delta_q, \delta_\pi) < \varepsilon$ or $n \geq N$
5. **return** $\boldsymbol{\theta}_n$, $\boldsymbol{w}_n$
```

Note that the policy $\pi_{\boldsymbol{w}}$ is being trained on a dataset $\mathcal{D}_\pi$ containing optimal actions computed with respect to an evolving Q-function. Specifically, at iteration $n$, we collect pairs $(s', a^*_{s'})$ where $a^*_{s'} = \arg\max_a q(s', a; \boldsymbol{\theta}_n)$. However, after updating to $\boldsymbol{\theta}_{n+1}$, these actions may no longer be optimal with respect to the new Q-function.

A natural approach to handle this staleness would be to maintain only the most recent optimization data. We could modify our procedure to keep a sliding window of $K$ iterations, where at iteration $n$, we only use data from iterations $\max(0, n-K)$ to $n$. This would be implemented by augmenting each entry in $\mathcal{D}_\pi$ with a timestamp:

$$
\mathcal{D}_\pi^{(n)} = \{(s', a^*_{s'}, t) \mid t \in \{n-K,\ldots,n\}\}
$$

where $t$ indicates the iteration at which the optimal action was computed. When fitting the policy network, we would then only use data points that are at most $K$ iterations old:

$$
\boldsymbol{w}_{n+1} \leftarrow \texttt{fit}(\{(s', a^*_{s'}) \mid (s', a^*_{s'}, t) \in \mathcal{D}_\pi^{(n)}, n-K \leq t \leq n\})
$$

This introduces a trade-off between using more data (larger $K$) versus using more recent, accurate data (smaller $K$). The choice of $K$ would depend on how quickly the Q-function evolves and the computational budget available for computing exact optimal actions.

The main limitation of this approach, beyond the out-of-distribution drift, is that it requires computing exact optimal actions via the solver for states in $\mathcal{B}_{\text{opt}}$. Can we reduce or eliminate this computational expense? As the policy improves at selecting actions, we can bootstrap from these increasingly better choices. Continuously amortizing these improving actions over time creates a virtuous cycle of self-improvement toward the optimal policy. However, this bootstrapping process requires careful management: moving too quickly can destabilize training.


# Deterministic Parametrized Policies 

In this section, we consider deterministic parametrized policies of the form $\pi_{\boldsymbol{w}}(s)$ which directly output an action given a state. This approach differs from stochastic policies that output probability distributions over actions, making it particularly suitable for continuous control problems where the optimal policy is often deterministic. Fitted Q-value methods can be naturally extended to simultaneously learn both the Q-function and such a deterministic policy.

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


## Neural Fitted Q-Iteration for Continuous Actions (NFQCA)

NFQCA {cite:p}`Hafner2011` extends the NFQI template from the [previous chapter](fqi.md) to handle continuous action spaces by replacing the $\max_{a'} q(s',a'; \boldsymbol{\theta})$ operator in the Bellman target with a parameterized policy $\pi_{\boldsymbol{w}}(s')$. This transforms fitted Q-iteration into an actor-critic method: the critic $q(s,a; \boldsymbol{\theta})$ evaluates state-action pairs via the standard regression step, while the actor $\pi_{\boldsymbol{w}}(s)$ provides actions by directly maximizing the learned Q-function.

The algorithm retains the two-level structure of NFQI: an outer loop performs approximate value iteration by computing Bellman targets, and an inner loop fits the Q-function to those targets. NFQCA adds a third component (policy improvement) that updates $\boldsymbol{w}$ to maximize the Q-function over states sampled from the dataset.

### From Discrete to Continuous Actions

Recall from the [FQI chapter](fqi.md) that NFQI computes Bellman targets using the hard max:

$$
y_{s,a} = r + \gamma \max_{a' \in \mathcal{A}} q(s',a'; \boldsymbol{\theta}_n)
$$

When $\mathcal{A}$ is finite and small, this max is computed by enumeration. When $\mathcal{A}$ is continuous or high-dimensional, enumeration is intractable. NFQCA replaces the max with a parameterized policy that approximately solves the maximization:

$$
y_{s,a} = r + \gamma q(s', \pi_{\boldsymbol{w}_n}(s'); \boldsymbol{\theta}_n)
$$

The policy $\pi_{\boldsymbol{w}}(s)$ acts as an **amortized optimizer**: instead of solving $\arg\max_{a'} q(s',a')$ from scratch at each state $s'$ during target computation, we train a neural network to output near-optimal actions directly. The term "amortized" refers to spreading the cost of optimization across training: we pay once to learn $\pi_{\boldsymbol{w}}$, then reuse it for all future target computations.

To train the policy, we maximize the expected Q-value under the distribution of states in the dataset. If we had access to the optimal Q-function $q^*$, we would solve:

$$
\max_{\boldsymbol{w}} \mathbb{E}_{s \sim \hat{P}_{\mathcal{D}}}[q^*(s, \pi_{\boldsymbol{w}}(s))]
$$

where $\hat{P}_{\mathcal{D}}$ is the empirical distribution over states induced by the offline dataset $\mathcal{D}$. In practice, we use the current Q-function approximation $q(s,a; \boldsymbol{\theta}_{n+1})$ after it has been fitted to the latest targets. The expectation is approximated by the sample average over states appearing in $\mathcal{D}$:

$$
\max_{\boldsymbol{w}} \frac{1}{|\mathcal{D}|} \sum_{(s,a,r,s') \in \mathcal{D}} q(s, \pi_{\boldsymbol{w}}(s); \boldsymbol{\theta}_{n+1})
$$

This policy improvement step runs after the Q-function has been updated, using the newly-fitted critic to guide the actor toward higher-value actions. Both the Q-function fitting and policy improvement use gradient-based optimization on the respective objectives.

```{prf:algorithm} Neural Fitted Q-Iteration with Continuous Actions (NFQCA)
:label: nfqca

**Input:** Dataset $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^N$, Q-function $q(s,a; \boldsymbol{\theta})$, policy $\pi_{\boldsymbol{w}}(s)$, discount factor $\gamma$, learning rates $\alpha_q, \alpha_\pi$, inner optimization steps $K_q, K_\pi$

**Output:** Q-function parameters $\boldsymbol{\theta}$, policy parameters $\boldsymbol{w}$

1. Initialize $\boldsymbol{\theta}_0$, $\boldsymbol{w}_0$
2. $n \leftarrow 0$
3. **repeat** $\quad$ // **Outer loop: Policy Evaluation and Improvement**
4. $\quad$ **// Construct regression dataset with policy-based Bellman targets**
5. $\quad$ $\mathcal{D}_n^{\text{fit}} \leftarrow \emptyset$
6. $\quad$ **for** each $(s,a,r,s') \in \mathcal{D}$ **do**
7. $\quad\quad$ $a' \leftarrow \pi_{\boldsymbol{w}_n}(s')$ $\quad$ // Actor selects action (replaces max)
8. $\quad\quad$ $y_{s,a} \leftarrow r + \gamma q(s', a'; \boldsymbol{\theta}_n)$ $\quad$ // Critic evaluates
9. $\quad\quad$ $\mathcal{D}_n^{\text{fit}} \leftarrow \mathcal{D}_n^{\text{fit}} \cup \{((s,a), y_{s,a})\}$
10. $\quad$ **// Policy evaluation: fit Q-function to targets**
11. $\quad$ $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}_q(\mathcal{D}_n^{\text{fit}}, \boldsymbol{\theta}_n, K_q, \alpha_q)$
12. $\quad$ **// Policy improvement: maximize Q-function over dataset states**
13. $\quad$ $\boldsymbol{w}_{n+1} \leftarrow \texttt{fit}_\pi(\mathcal{D}, \boldsymbol{w}_n, \boldsymbol{\theta}_{n+1}, K_\pi, \alpha_\pi)$
14. $\quad$ $n \leftarrow n + 1$
15. **until** convergence or $n \geq n_{\max}$
16. **return** $\boldsymbol{\theta}_n$, $\boldsymbol{w}_n$

**where:**
- $\texttt{fit}_q(\mathcal{D}_n^{\text{fit}}, \boldsymbol{\theta}_n, K_q, \alpha_q)$ runs $K_q$ gradient steps minimizing $\frac{1}{|\mathcal{D}_n^{\text{fit}}|} \sum_{((s,a), y) \in \mathcal{D}_n^{\text{fit}}} (q(s,a; \boldsymbol{\theta}) - y)^2$, warm starting from $\boldsymbol{\theta}_n$
- $\texttt{fit}_\pi(\mathcal{D}, \boldsymbol{w}_n, \boldsymbol{\theta}_{n+1}, K_\pi, \alpha_\pi)$ runs $K_\pi$ gradient steps maximizing $\frac{1}{|\mathcal{D}|} \sum_{(s,a,r,s') \in \mathcal{D}} q(s, \pi_{\boldsymbol{w}}(s); \boldsymbol{\theta}_{n+1})$, warm starting from $\boldsymbol{w}_n$
```

The algorithm structure mirrors NFQI (Algorithm {prf:ref}`fitted-q-iteration-batch` in the [FQI chapter](fqi.md)) with two extensions. First, target computation (line 7-8) replaces the discrete max with a policy network call $\pi_{\boldsymbol{w}_n}(s')$, making the Bellman operator tractable for continuous actions. Second, after fitting the Q-function (line 11), we add a policy improvement step (line 13) that updates $\boldsymbol{w}$ to maximize the Q-function evaluated at policy-generated actions over states in the dataset.

Both `fit` operations use gradient descent with warm starting, consistent with the NFQI template. The Q-function minimizes squared Bellman error using targets computed with the current policy. The policy maximizes the Q-function via gradient ascent on the composition $q(s, \pi_{\boldsymbol{w}}(s); \boldsymbol{\theta}_{n+1})$, which is differentiable end-to-end when both networks are differentiable. The gradient with respect to $\boldsymbol{w}$ is:

$$
\nabla_{\boldsymbol{w}} q(s, \pi_{\boldsymbol{w}}(s); \boldsymbol{\theta}) = \nabla_a q(s, a; \boldsymbol{\theta})\Big|_{a=\pi_{\boldsymbol{w}}(s)} \cdot \nabla_{\boldsymbol{w}} \pi_{\boldsymbol{w}}(s)
$$

computed via the chain rule (backpropagation through the actor into the critic). Modern automatic differentiation libraries handle this composition automatically.

## Euler Equation Methods: Approximating Policies Instead of Values

The weighted residual framework applies to any functional equation arising from an MDP. So far we have applied it to the Bellman equation to approximate the value function. An alternative is to approximate the **optimal policy** directly by enforcing first-order optimality conditions {cite:p}`Judd1992,Rust1996,ndp`.

Consider a control problem with continuous states and actions, deterministic dynamics $s' = f(s,a)$, and reward $r(s,a)$, both continuously differentiable. For each state $s$, the optimal action $\pi^*(s)$ satisfies the first-order condition

$$
\frac{\partial r(s,a)}{\partial a}\Big|_{a=\pi^*(s)}
+
\gamma\, \frac{\partial v^*(s')}{\partial s'}\Big|_{s'=f(s,\pi^*(s))}\,
\frac{\partial f(s,a)}{\partial a}\Big|_{a=\pi^*(s)}
=
0.
$$ (Euler-raw)

This involves the unknown value gradient $\partial v^*/\partial s'$ evaluated at the next state. In a general MDP, one would need to solve jointly for both the policy and the value-function derivatives.

### The Euler class

Many control problems have additional structure that eliminates $\partial v^*/\partial s$ from the first-order condition. Rust formalizes such problems as an **Euler class** {cite:p}`Rust1996`: the state $s$ can be written as $(s_1, s_2)$ where $s_1$ is controlled (inventory, battery charge, water level) and $s_2$ evolves independently (demand, weather, prices). The transition law factors as

$$
p(s_1', s_2' \mid s_1, s_2, a)
=
\mathbf{1}\{s_1' = f(s_1, a, s_2, s_2')\} \cdot q(s_2' \mid s_2),
$$

with $f$ continuously differentiable. The **Euler-class condition** requires a function $h(s_1, a, s_2)$ such that

$$
\frac{\partial f(s_1, a, s_2, s_2')}{\partial s_1}\Big|_{(s_1, a, s_2, s_2')}
=
\frac{\partial f(s_1, a, s_2, s_2')}{\partial a}\Big|_{(s_1, a, s_2, s_2')} \cdot h(s_1, a, s_2).
$$ (EC)

In scalar problems, this means the derivative with respect to the controlled state is proportional to the derivative with respect to the action. For affine dynamics $s_1' = \alpha s_1 + \beta a + \delta$ (where $\alpha, \beta, \delta$ may depend on $s_2, s_2'$), we have $\partial f/\partial s_1 = \alpha$ and $\partial f/\partial a = \beta$, so the condition holds with $h = \alpha/\beta$. This covers inventory ($s_1' = \alpha s_1 + a - D(s_2')$), energy storage ($s_1' = \eta_{\mathrm{ret}} s_1 + \eta_{\mathrm{ch}} a$), reservoir ($s_1' = \alpha s_1 + I(s_2') - a$), and thermal models ($s_1' = \alpha(s_2) s_1 + \beta(s_2) a + \delta(s_2)$).

Under this condition, an envelope formula expresses $\partial v^*(s_1, s_2)/\partial s_1$ purely in terms of $r$, $h$, and $\pi^*$:

$$
\frac{\partial v^*(s_1, s_2)}{\partial s_1}\Big|_{(s_1, s_2)}
=
\frac{\partial r(s_1, s_2, \pi^*(s_1, s_2))}{\partial s_1}\Big|_{(s_1, s_2)}
+
\frac{\partial r(s_1, s_2, \pi^*(s_1, s_2))}{\partial a}\Big|_{(s_1, s_2)} \cdot h(s_1, \pi^*(s_1, s_2), s_2).
$$

Substituting into the first-order condition eliminates $v^*$ entirely, yielding an equation $\mathcal{E}(\pi)(s_1, s_2) = 0$ depending only on primitives $r, f, h, q$ and the policy $\pi$. This transforms a coupled system (policy + value) into a closed functional equation in the policy alone.


### Discretization

For a parameterized policy $\pi_{\boldsymbol{\theta}}(s)$, we discretize the Euler equation using weighted residuals. With collocation at points $s_1, \dots, s_N$, we solve

$$
G(\boldsymbol{\theta}) := \begin{bmatrix}
\mathcal{E}(\pi_{\boldsymbol{\theta}})(s_1) \\
\vdots \\
\mathcal{E}(\pi_{\boldsymbol{\theta}})(s_N)
\end{bmatrix} = 0.
$$

With Galerkin projection using test functions $\{\varphi_i\}_{i=1}^M$ and weighting measure $\mu$, we solve

$$
\int \varphi_i(s) \mathcal{E}(\pi_{\boldsymbol{\theta}})(s) \mu(ds) = 0,
\quad i = 1,\dots,M.
$$

The mapping $G$ is nonlinear in $\boldsymbol{\theta}$, requiring Newton-type methods or other root-finding schemes. Unlike the Bellman operator, this operator is not a contraction, so convergence guarantees are problem-dependent.

## Deep Deterministic Policy Gradient (DDPG)

We now extend NFQCA to the online setting with evolving replay buffers, mirroring how DQN extended NFQI in the [FQI chapter](fqi.md). Just as DQN allowed $\mathcal{B}_t$ and $\hat{P}_{\mathcal{B}_t}$ to evolve during learning instead of using a fixed offline dataset, DDPG {cite:p}`lillicrap2015continuous` collects new transitions during training and stores them in a circular replay buffer.

Like DQN, DDPG uses the flattened FQI structure with target networks. But where DQN maintains a single target network $\boldsymbol{\theta}_{\text{target}}$ for the Q-function, DDPG maintains **two** target networks: one for the critic $\boldsymbol{\theta}_{\text{target}}$ and one for the actor $\boldsymbol{w}_{\text{target}}$. Both are updated periodically (every $K$ steps) to mark outer-iteration boundaries, following the same nested-to-flattened transformation shown for DQN.

The online network now plays a triple role in DDPG: (1) the parameters being actively trained ($\boldsymbol{\theta}_t$ for critic, $\boldsymbol{w}_t$ for actor), (2) the policy used to collect new data, and (3) the gradient source for policy improvement. The target networks serve only one purpose: computing stable Bellman targets.

### Exploration via Action Noise

Since the policy $\pi_{\boldsymbol{w}}(s)$ is deterministic, exploration requires adding noise to actions during data collection:

$$
a = \pi_{\boldsymbol{w}_t}(s) + \eta_t
$$

where $\eta_t$ is exploration noise. The original DDPG paper used an Ornstein-Uhlenbeck (OU) process, which generates temporally correlated noise through the discretized stochastic differential equation:

$$
\eta_{t+1} = \eta_t + \theta(\mu - \eta_t)\Delta t + \sigma\sqrt{\Delta t}\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0,1)
$$

where $\mu$ is the long-term mean (typically 0), $\theta$ controls the strength of mean reversion, $\sigma$ scales the random fluctuations, and $\Delta t$ is the time step. The term $\theta(\mu - \eta_t)\Delta t$ acts like damped motion through a viscous fluid: when $\eta_t$ deviates from $\mu$, this force pulls it back smoothly without oscillation. The random term $\sigma\sqrt{\Delta t}\epsilon_t$ adds perturbations, creating noise that wanders but is gently pulled back toward $\mu$. This temporal correlation produces smoother exploration trajectories than independent Gaussian noise.

However, later work (including TD3, discussed below) found that simple uncorrelated Gaussian noise $\eta_t \sim \mathcal{N}(0, \sigma^2)$ works equally well and is easier to tune. The exploration mechanism is orthogonal to the core algorithmic structure.

```{prf:algorithm} Deep Deterministic Policy Gradient (DDPG)
:label: ddpg

**Input:** MDP $(S, \mathcal{A}, P, R, \gamma)$, Q-network $q(s,a; \boldsymbol{\theta})$, policy network $\pi_{\boldsymbol{w}}(s)$, learning rates $\alpha_q, \alpha_\pi$, target update frequency $K$, replay buffer capacity $B$, mini-batch size $b$, exploration noise $\eta$

**Output:** Q-function parameters $\boldsymbol{\theta}$, policy parameters $\boldsymbol{w}$

1. Initialize $\boldsymbol{\theta}_0$, $\boldsymbol{w}_0$ randomly
2. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_0$, $\boldsymbol{w}_{\text{target}} \leftarrow \boldsymbol{w}_0$
3. Initialize replay buffer $\mathcal{B}$ with capacity $B$
4. Initialize exploration noise process $\eta$ (e.g., OU process or Gaussian)
5. $t \leftarrow 0$
6. **while** training **do**
    1. Observe current state $s$
    2. **// Use online network to collect data with exploration noise**
    3. Select action: $a \leftarrow \pi_{\boldsymbol{w}_t}(s) + \eta_t$
    4. Execute $a$, observe reward $r$ and next state $s'$
    5. Store $(s,a,r,s')$ in $\mathcal{B}$, replacing oldest if full
    6. Sample mini-batch of $b$ transitions $\{(s_i,a_i,r_i,s_i')\}_{i=1}^b$ from $\mathcal{B}$
    7. **for** each sampled transition $(s_i,a_i,r_i,s_i')$ **do**
        1. $y_i \leftarrow r_i + \gamma q(s'_i, \pi_{\boldsymbol{w}_{\text{target}}}(s'_i); \boldsymbol{\theta}_{\text{target}})$ $\quad$ // Actor target selects, critic target evaluates
    8. $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha_q \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_{i=1}^b (q(s_i,a_i;\boldsymbol{\theta}_t) - y_i)^2$
    9. $\boldsymbol{w}_{t+1} \leftarrow \boldsymbol{w}_t + \alpha_\pi \frac{1}{b}\sum_{i=1}^b \nabla_a q(s_i,a;\boldsymbol{\theta}_{t+1})\Big|_{a=\pi_{\boldsymbol{w}_t}(s_i)} \cdot \nabla_{\boldsymbol{w}} \pi_{\boldsymbol{w}_t}(s_i)$
    10. **if** $t \bmod K = 0$ **then**
        1. $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_t$
        2. $\boldsymbol{w}_{\text{target}} \leftarrow \boldsymbol{w}_t$
    11. $t \leftarrow t + 1$
7. **return** $\boldsymbol{\theta}_t$, $\boldsymbol{w}_t$
```

The algorithm structure parallels DQN (Algorithm {prf:ref}`dqn` in the [FQI chapter](fqi.md)) with the continuous-action extensions from NFQCA. Lines 1-5 initialize both networks and their targets, following the same pattern as DQN but with an additional actor network. Line 3 uses the online actor with exploration noise for data collection, replacing DQN's $\varepsilon$-greedy selection. Line 7 computes targets using both target networks: the actor target $\pi_{\boldsymbol{w}_{\text{target}}}(s'_i)$ selects the next action, the critic target $q(\cdot; \boldsymbol{\theta}_{\text{target}})$ evaluates it. This replaces the $\max_{a'}$ operator in DQN. Lines 8-9 update both networks: critic via TD error minimization, actor via policy gradient through the updated critic. Line 10 performs periodic hard updates every $K$ steps, marking outer-iteration boundaries.

The policy gradient in line 9 uses the chain rule to backpropagate through the actor-critic composition:

$$
\nabla_{\boldsymbol{w}} q(s, \pi_{\boldsymbol{w}}(s); \boldsymbol{\theta}) = \nabla_a q(s,a; \boldsymbol{\theta})\Big|_{a=\pi_{\boldsymbol{w}}(s)} \cdot \nabla_{\boldsymbol{w}} \pi_{\boldsymbol{w}}(s)
$$

This is identical to the NFQCA gradient, but now computed on mini-batches sampled from an evolving replay buffer rather than a fixed offline dataset. The critic gradient $\nabla_a q(s,a; \boldsymbol{\theta})$ at the policy-generated action provides the direction of steepest ascent in Q-value space, weighted by how sensitive the policy output is to its parameters via $\nabla_{\boldsymbol{w}} \pi_{\boldsymbol{w}}(s)$.

## Twin Delayed Deep Deterministic Policy Gradient (TD3)

DDPG inherits the overestimation bias from DQN's use of the max operator in Bellman targets. TD3 {cite:p}`fujimoto2018addressing` addresses this through three modifications to the DDPG template, following similar principles to Double DQN but adapted for continuous actions and taking a more conservative approach.

### Twin Q-Networks and the Minimum Operator

Recall from the [Monte Carlo chapter](montecarlo.md) that overestimation arises when we use the same noisy estimate both to select which action looks best and to evaluate that action. Double Q-learning breaks this coupling by maintaining two independent estimators with noise terms $\varepsilon^{(1)}_a$ and $\varepsilon^{(2)}_a$:

$$
a^\star = \arg\max_{a} \left\{r(s,a) + \gamma \hat{\mu}^{(1)}_N(s,a)\right\}, \quad Y = r(s,a^\star) + \gamma \hat{\mu}^{(2)}_N(s,a^\star).
$$

When $\varepsilon^{(1)}$ and $\varepsilon^{(2)}$ are independent, the tower property of conditional expectation gives $\mathbb{E}[\varepsilon^{(2)}_{a^\star} \mid a^\star] = \mathbb{E}[\varepsilon^{(2)}_{a^\star}] = 0$ because $a^\star$ (determined by $\varepsilon^{(1)}$) is independent of $\varepsilon^{(2)}$. This eliminates **evaluation bias**: we no longer use the same positive noise that selected an action to also inflate its value. By conditioning on the selected action and then taking expectations over the independent evaluation noise, the bias in the evaluation term vanishes.

Double DQN (Algorithm {prf:ref}`double-dqn`) implements this principle in the discrete action setting by using the online network $\boldsymbol{\theta}_t$ for selection ($a^*_i \leftarrow \arg\max_{a'} q(s_i',a'; \boldsymbol{\theta}_t)$) and the target network $\boldsymbol{\theta}_{\text{target}}$ for evaluation ($y_i \leftarrow r_i + \gamma q(s_i',a^*_i; \boldsymbol{\theta}_{\text{target}})$). Since these networks experience different training noise, their errors are approximately independent, achieving the independence condition needed to eliminate evaluation bias. However, **selection bias** remains: the argmax still picks actions that received positive noise in the selection network, so $\mathbb{E}_{\varepsilon^{(1)}}[\mu(s,a^\star)] \ge \max_a \mu(s,a)$.

TD3 takes a more conservative approach. Instead of decoupling selection from evaluation, TD3 maintains **twin Q-networks** $q^A(s,a; \boldsymbol{\theta}^A)$ and $q^B(s,a; \boldsymbol{\theta}^B)$ trained on the same data with different random initializations. When computing targets, TD3 uses the target policy $\pi_{\boldsymbol{w}_{\text{target}}}(s')$ to select actions (no maximization over a discrete set), then takes the minimum of the two Q-networks' evaluations:

$$
y_i = r_i + \gamma \min\left(q^A(s'_i, \tilde{a}_i; \boldsymbol{\theta}^A_{\text{target}}), q^B(s'_i, \tilde{a}_i; \boldsymbol{\theta}^B_{\text{target}})\right)
$$

where $\tilde{a}_i = \pi_{\boldsymbol{w}_{\text{target}}}(s'_i)$. This minimum operation provides a pessimistic estimate: if the two Q-networks have independent errors $q^A(s',a) = q^*(s',a) + \varepsilon^A$ and $q^B(s',a) = q^*(s',a) + \varepsilon^B$, then $\mathbb{E}[\min(q^A, q^B)] \le q^*(s',a)$, producing systematic underestimation rather than overestimation.

The connection to the conditional independence framework is subtle but important. While Double DQN uses independence to eliminate bias in expectation (one network selects, another evaluates), TD3 uses independence to construct a deliberate lower bound. Both approaches rely on maintaining two Q-functions with partially decorrelated errors, achieved through different initializations and stochastic gradients during training, but they aggregate these functions differently. Double DQN's decoupling targets unbiased estimation by breaking the correlation between selection and evaluation noise. TD3's minimum operation targets robust estimation by taking the most pessimistic view when the two networks disagree.

This trade-off between bias and robustness is deliberate. In actor-critic methods, the policy gradient pushes toward actions with high Q-values. Overestimation is particularly harmful because it can lead the policy to exploit erroneous high-value regions. Underestimation is generally safer: the policy may ignore some good actions, but it will not be misled into pursuing actions that only appear valuable due to approximation error. The minimum operation implements a "trust the pessimist" principle that complements the policy optimization objective.

TD3 also introduces two additional modifications beyond the clipped double Q-learning. First, target policy smoothing adds clipped noise to the target policy's actions when computing targets: $\tilde{a} = \pi_{\boldsymbol{w}_{\text{target}}}(s') + \text{clip}(\varepsilon, -c, c)$. This regularization prevents the policy from exploiting narrow peaks in the Q-function approximation error by averaging over nearby actions. Second, delayed policy updates change the actor update frequency: the actor updates every $d$ steps instead of every step. This reduces per-update error by letting the critics converge before the actor adapts to them.

TD3 also replaces DDPG's hard target updates with **exponential moving average (EMA)** updates, following the smooth update scheme from Algorithm {prf:ref}`nfqi-flattened-ema` in the [FQI chapter](fqi.md). Instead of copying $\boldsymbol{\theta}_{\text{target}} \leftarrow \boldsymbol{\theta}_t$ every $K$ steps, EMA smoothly tracks the online network: $\boldsymbol{\theta}_{\text{target}} \leftarrow \tau \boldsymbol{\theta}_t + (1-\tau)\boldsymbol{\theta}_{\text{target}}$ at every update. For small $\tau \in [0.001, 0.01]$, the target lags behind the online network by roughly $1/\tau$ steps, providing smoother learning dynamics.

```{prf:algorithm} Twin Delayed Deep Deterministic Policy Gradient (TD3)
:label: td3

**Input:** MDP $(S, \mathcal{A}, P, R, \gamma)$, twin Q-networks $q^A(s,a; \boldsymbol{\theta}^A)$, $q^B(s,a; \boldsymbol{\theta}^B)$, policy network $\pi_{\boldsymbol{w}}(s)$, learning rates $\alpha_q, \alpha_\pi$, replay buffer capacity $B$, mini-batch size $b$, policy delay $d$, EMA rate $\tau$, target noise $\sigma$, noise clip $c$, exploration noise $\sigma_{\text{explore}}$

**Output:** Twin Q-function parameters $\boldsymbol{\theta}^A, \boldsymbol{\theta}^B$, policy parameters $\boldsymbol{w}$

1. Initialize $\boldsymbol{\theta}^A_0$, $\boldsymbol{\theta}^B_0$, $\boldsymbol{w}_0$ randomly
2. $\boldsymbol{\theta}^A_{\text{target}} \leftarrow \boldsymbol{\theta}^A_0$, $\boldsymbol{\theta}^B_{\text{target}} \leftarrow \boldsymbol{\theta}^B_0$, $\boldsymbol{w}_{\text{target}} \leftarrow \boldsymbol{w}_0$
3. Initialize replay buffer $\mathcal{B}$ with capacity $B$
4. $t \leftarrow 0$
5. **while** training **do**
    1. Observe current state $s$
    2. **// Data collection with exploration noise**
    3. Select action: $a \leftarrow \pi_{\boldsymbol{w}_t}(s) + \varepsilon$, where $\varepsilon \sim \mathcal{N}(0, \sigma_{\text{explore}}^2)$
    4. Execute $a$, observe reward $r$ and next state $s'$
    5. Store $(s,a,r,s')$ in $\mathcal{B}$, replacing oldest if full
    6. Sample mini-batch of $b$ transitions $\{(s_i,a_i,r_i,s_i')\}_{i=1}^b$ from $\mathcal{B}$
    7. **// Compute targets with clipped double Q-learning**
    8. **for** each sampled transition $(s_i,a_i,r_i,s_i')$ **do**
        1. $\tilde{a}_i \leftarrow \pi_{\boldsymbol{w}_{\text{target}}}(s'_i) + \text{clip}(\varepsilon_i, -c, c)$, $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ $\quad$ // Target policy smoothing
        2. $y_i \leftarrow r_i + \gamma \color{blue}{\min\big(q^A(s'_i, \tilde{a}_i; \boldsymbol{\theta}^A_{\text{target}}), q^B(s'_i, \tilde{a}_i; \boldsymbol{\theta}^B_{\text{target}})\big)}$ $\quad$ // Minimum of twin targets
    9. **// Update both Q-networks toward same targets**
    10. $\boldsymbol{\theta}^A_{t+1} \leftarrow \boldsymbol{\theta}^A_t - \alpha_q \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_{i=1}^b(q^A(s_i,a_i;\boldsymbol{\theta}^A_t) - y_i)^2$
    11. $\boldsymbol{\theta}^B_{t+1} \leftarrow \boldsymbol{\theta}^B_t - \alpha_q \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_{i=1}^b(q^B(s_i,a_i;\boldsymbol{\theta}^B_t) - y_i)^2$
    12. **// Delayed policy update and target network updates**
    13. **if** $t \bmod d = 0$ **then**
        1. $\boldsymbol{w}_{t+1} \leftarrow \boldsymbol{w}_t + \alpha_\pi \frac{1}{b}\sum_{i=1}^b \nabla_a q^A(s_i,a;\boldsymbol{\theta}^A_{t+1})\Big|_{a=\pi_{\boldsymbol{w}_t}(s_i)} \cdot \nabla_{\boldsymbol{w}} \pi_{\boldsymbol{w}_t}(s_i)$
        2. $\boldsymbol{\theta}^A_{\text{target}} \leftarrow \tau\boldsymbol{\theta}^A_{t+1} + (1-\tau)\boldsymbol{\theta}^A_{\text{target}}$ (EMA update)
        3. $\boldsymbol{\theta}^B_{\text{target}} \leftarrow \tau\boldsymbol{\theta}^B_{t+1} + (1-\tau)\boldsymbol{\theta}^B_{\text{target}}$
        4. $\boldsymbol{w}_{\text{target}} \leftarrow \tau\boldsymbol{w}_{t+1} + (1-\tau)\boldsymbol{w}_{\text{target}}$
    14. $t \leftarrow t + 1$
6. **return** $\boldsymbol{\theta}^A_t$, $\boldsymbol{\theta}^B_t$, $\boldsymbol{w}_t$
```

The algorithm structure parallels Double DQN but with continuous actions. Lines 8.1-8.2 implement clipped double Q-learning: smoothing adds noise to target actions (preventing exploitation of Q-function artifacts), and the min operation (highlighted in blue) provides pessimistic value estimates. Both critics update toward the same shared target (lines 10-11), but their different initializations and stochastic gradient noise keep their errors partially decorrelated, following the same principle underlying Double DQN's independence assumption. Line 13 gates policy updates to every $d$ steps (typically $d=2$), and lines 13.2-13.4 use EMA updates following Algorithm {prf:ref}`nfqi-flattened-ema`.

TD3 simplifies exploration by replacing DDPG's Ornstein-Uhlenbeck process with uncorrelated Gaussian noise $\varepsilon \sim \mathcal{N}(0, \sigma_{\text{explore}}^2)$ (line 5.3). This eliminates the need to tune multiple OU parameters while providing equally effective exploration.

# Soft Actor-Critic

DDPG and TD3 address continuous actions by learning a deterministic policy that amortizes the $\arg\max_a q(s,a)$ operation. But deterministic policies have a fundamental limitation: they require external exploration noise (Gaussian perturbations in TD3) and can converge to suboptimal deterministic behaviors without adequate coverage of the state-action space.

The [smoothing chapter](smoothing.md) presents an alternative: entropy-regularized MDPs, where the agent maximizes expected return plus a bonus for policy randomness. This yields stochastic policies with exploration built into the objective itself. The smooth Bellman operator replaces the hard max with a soft-max:

$$
v^*(s) = \frac{1}{\beta} \log \sum_{a \in \mathcal{A}} \exp\left(\beta \cdot q^*(s,a)\right)
$$

where $\beta = 1/\alpha$ is the inverse temperature and $\alpha$ is the entropy regularization weight. For finite action spaces, this log-sum-exp is easy to compute. But for continuous actions $\mathcal{A} \subset \mathbb{R}^m$, the sum becomes an integral:

$$
v^*(s) = \frac{1}{\beta} \log \int_{\mathcal{A}} \exp\left(\beta \cdot q^*(s,a)\right) da
$$

This integral is intractable. We face an infinite-dimensional sum over the continuous action space. The very smoothness that gives us stochastic policies creates a new computational barrier, distinct from but analogous to the $\arg\max$ problem in standard FQI.

## From Intractable Integral to Tractable Expectation

Soft actor-critic (SAC) {cite:p}`haarnoja2018soft,haarnoja2018sacapplications` exploits an equivalence between the intractable integral and an expectation. The optimal policy under entropy regularization is the Boltzmann distribution:

$$
\pi^*(a|s) = \frac{\exp(\beta \cdot q^*(s,a))}{\int_{\mathcal{A}} \exp(\beta \cdot q^*(s,a')) da'} \propto \exp(\beta \cdot q^*(s,a))
$$

Under this policy, the soft value function becomes:

$$
v^*(s) = \mathbb{E}_{a \sim \pi^*(\cdot|s)}\left[q^*(s,a) - \alpha \log \pi^*(a|s)\right]
$$

We have converted an intractable integral into an expectation that we can estimate by sampling. The catch: we need samples from $\pi^*$, which depends on the $q^*$ we are trying to learn.

SAC uses the same policy amortization strategy as DDPG: learn a parametric policy $\pi_{\boldsymbol{\phi}}$ that approximates the optimal stochastic policy (the Boltzmann distribution). The policy enables fast action selection through a single forward pass rather than solving an optimization problem. Exploration comes from the policy's stochasticity rather than from external noise.

## Bootstrap Targets via Single-Sample Estimation

With a learned policy $\pi_{\boldsymbol{\phi}}$, we can compute Q-function bootstrap targets. For a transition $(s,a,r,s')$, we need the soft value at $s'$:

$$
v(s') = \mathbb{E}_{a' \sim \pi_{\boldsymbol{\phi}}(\cdot|s')}\left[q(s',a') - \alpha \log \pi_{\boldsymbol{\phi}}(a'|s')\right]
$$

SAC estimates this with a single Monte Carlo sample: draw $\tilde{a}' \sim \pi_{\boldsymbol{\phi}}(\cdot|s')$ and approximate:

$$
y = r + \gamma \left[\min_{j=1,2} q^j_{\boldsymbol{\theta}_{\text{target}}}(s', \tilde{a}') - \alpha \log \pi_{\boldsymbol{\phi}}(\tilde{a}'|s')\right]
$$

The minimum over twin Q-networks applies the clipped double-Q trick from TD3. This single-sample approach is computationally efficient: each target requires just one policy sample and two Q-network evaluations.

```{admonition} Historical Note: The V-Network in Original SAC
:class: dropdown

The original SAC paper {cite:p}`haarnoja2018soft` introduced a separate value network $v_\psi(s)$ trained to predict the entropy-adjusted expectation, amortizing the soft value computation into a single forward pass. Bootstrap targets then used $y = r + \gamma v_{\psi_{\text{target}}}(s')$.

However, the follow-up paper {cite:p}`haarnoja2018sacapplications` showed this V-network is redundant: the single-sample estimate works just as well while simplifying the architecture. All modern implementations — OpenAI Spinning Up, Stable Baselines3, CleanRL — omit the V-network. We present this simplified version throughout.
```

## Learning the Policy: Matching the Boltzmann Distribution

The Q-network update assumes a policy $\pi_{\boldsymbol{\phi}}$ that approximates the Boltzmann distribution $\pi^*(a|s) \propto \exp(\beta \cdot q^*(s,a))$. Training such a policy presents a problem: the Boltzmann distribution requires the partition function $Z(s) = \int_{\mathcal{A}} \exp(\beta \cdot q(s,a))da$, the very integral we are trying to avoid. SAC sidesteps this by minimizing the KL divergence from the policy to the (unnormalized) Boltzmann distribution:

$$
\min_{\boldsymbol{\phi}} \mathbb{E}_{s \sim \mathcal{D}}\left[D_{KL}\left(\pi_{\boldsymbol{\phi}}(\cdot|s) \| \frac{\exp(\beta \cdot q_{\boldsymbol{\theta}}(s,\cdot))}{Z_{\boldsymbol{\theta}}(s)}\right)\right]
$$

Since $\log Z_{\boldsymbol{\theta}}(s)$ does not depend on $\boldsymbol{\phi}$, this reduces to:

$$
\min_{\boldsymbol{\phi}} \mathbb{E}_{s \sim \mathcal{D}}\mathbb{E}_{a \sim \pi_{\boldsymbol{\phi}}(\cdot|s)}\left[\alpha \log \pi_{\boldsymbol{\phi}}(a|s) - q_{\boldsymbol{\theta}}(s,a)\right]
$$

This pushes probability toward high Q-value actions while the $\log \pi_{\boldsymbol{\phi}}$ term penalizes concentrating probability mass, maintaining entropy. The entropy bonus comes from the KL divergence structure rather than from an explicit regularization term.

To estimate gradients of this objective, we face a technical problem: the policy parameters $\boldsymbol{\phi}$ appear in the sampling distribution $\pi_{\boldsymbol{\phi}}$, making $\nabla_{\boldsymbol{\phi}} \mathbb{E}_{a \sim \pi_{\boldsymbol{\phi}}}[\cdot]$ difficult to compute. SAC uses a Gaussian policy $\pi_{\boldsymbol{\phi}}(a|s) = \mathcal{N}(\mu_{\boldsymbol{\phi}}(s), \sigma_{\boldsymbol{\phi}}(s)^2)$ with the reparameterization trick. Express samples as a deterministic function of parameters and independent noise:

$$
a = f_{\boldsymbol{\phi}}(s, \epsilon) = \mu_{\boldsymbol{\phi}}(s) + \sigma_{\boldsymbol{\phi}}(s) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

This moves $\boldsymbol{\phi}$ out of the sampling distribution and into the integrand:

$$
\min_{\boldsymbol{\phi}} \mathbb{E}_{s \sim \mathcal{D}}\mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}\left[\alpha \log \pi_{\boldsymbol{\phi}}(f_{\boldsymbol{\phi}}(s,\epsilon)|s) - q_{\boldsymbol{\theta}}(s,f_{\boldsymbol{\phi}}(s,\epsilon))\right]
$$

We can now differentiate through $f_{\boldsymbol{\phi}}$ and the Q-network, as DDPG differentiates through a deterministic policy. SAC extends this by sampling noise $\epsilon$ at each gradient step rather than outputting a single deterministic action.


```{prf:algorithm} Soft Actor-Critic (SAC)
:label: sac

**Input:** MDP $(S, \mathcal{A}, P, R, \gamma)$, twin Q-networks $q^1(s,a; \boldsymbol{\theta}^1), q^2(s,a; \boldsymbol{\theta}^2)$, policy $\pi_{\boldsymbol{\phi}}$, learning rates $\alpha_q, \alpha_\pi$, replay buffer capacity $B$, mini-batch size $b$, EMA rate $\tau$, entropy weight $\alpha$

**Output:** Q-function parameters $\boldsymbol{\theta}^1, \boldsymbol{\theta}^2$, policy parameters $\boldsymbol{\phi}$

1. Initialize $\boldsymbol{\theta}^1_0, \boldsymbol{\theta}^2_0, \boldsymbol{\phi}_0$ randomly
2. $\boldsymbol{\theta}^1_{\text{target}} \leftarrow \boldsymbol{\theta}^1_0$, $\boldsymbol{\theta}^2_{\text{target}} \leftarrow \boldsymbol{\theta}^2_0$
3. Initialize replay buffer $\mathcal{B}$ with capacity $B$
4. $t \leftarrow 0$
5. **while** training **do**
    1. Observe state $s$
    2. Sample action: $a \sim \pi_{\boldsymbol{\phi}_t}(\cdot|s)$ $\quad$ // Stochastic policy provides exploration
    3. Execute $a$, observe reward $r$ and next state $s'$
    4. Store $(s,a,r,s')$ in $\mathcal{B}$, replacing oldest if full
    5. Sample mini-batch $\{(s_i,a_i,r_i,s_i')\}_{i=1}^b$ from $\mathcal{B}$
    6. **// Update Q-networks: bootstrap using single-sample soft value estimate**
    7. **for** each $s'_i$ **do** sample $\tilde{a}'_i \sim \pi_{\boldsymbol{\phi}_t}(\cdot|s'_i)$
    8. $y_i \leftarrow r_i + \gamma \left[\min(q^1(s'_i, \tilde{a}'_i; \boldsymbol{\theta}^1_{\text{target}}), q^2(s'_i, \tilde{a}'_i; \boldsymbol{\theta}^2_{\text{target}})) - \alpha \log \pi_{\boldsymbol{\phi}_t}(\tilde{a}'_i|s'_i)\right]$
    9. $\boldsymbol{\theta}^1_{t+1} \leftarrow \boldsymbol{\theta}^1_t - \alpha_q \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_{i=1}^b(q^1(s_i,a_i;\boldsymbol{\theta}^1_t) - y_i)^2$
    10. $\boldsymbol{\theta}^2_{t+1} \leftarrow \boldsymbol{\theta}^2_t - \alpha_q \nabla_{\boldsymbol{\theta}} \frac{1}{b}\sum_{i=1}^b(q^2(s_i,a_i;\boldsymbol{\theta}^2_t) - y_i)^2$
    11. **// Update policy: minimize KL to Boltzmann distribution**
    12. **for** each $s_i$ **do** sample $\epsilon_i \sim \mathcal{N}(0,I)$ and compute $\hat{a}_i = f_{\boldsymbol{\phi}_t}(s_i, \epsilon_i)$ $\quad$ // Reparameterization
    13. $\boldsymbol{\phi}_{t+1} \leftarrow \boldsymbol{\phi}_t - \alpha_\pi \nabla_{\boldsymbol{\phi}} \frac{1}{b}\sum_{i=1}^b \left[\alpha \log \pi_{\boldsymbol{\phi}_t}(\hat{a}_i|s_i) - \min_{j=1,2} q^j(s_i, \hat{a}_i; \boldsymbol{\theta}^j_{t+1})\right]$
    14. **// EMA update for Q-network targets**
    15. $\boldsymbol{\theta}^1_{\text{target}} \leftarrow \tau \boldsymbol{\theta}^1_{t+1} + (1-\tau)\boldsymbol{\theta}^1_{\text{target}}$
    16. $\boldsymbol{\theta}^2_{\text{target}} \leftarrow \tau \boldsymbol{\theta}^2_{t+1} + (1-\tau)\boldsymbol{\theta}^2_{\text{target}}$
    17. $t \leftarrow t + 1$
6. **return** $\boldsymbol{\theta}^1_t, \boldsymbol{\theta}^2_t, \boldsymbol{\phi}_t$
```

The algorithm interleaves three updates. The Q-networks (lines 7-10) follow fitted Q-iteration with the soft Bellman target: sample a next action $\tilde{a}'_i$ from the current policy, compute the entropy-adjusted target $y_i = r_i + \gamma[\min_j q^j_{\text{target}}(s'_i, \tilde{a}'_i) - \alpha \log \pi_{\boldsymbol{\phi}}(\tilde{a}'_i|s'_i)]$, and minimize squared error. The minimum over twin Q-networks mitigates overestimation as in TD3. The policy (lines 12-13) updates to match the Boltzmann distribution induced by the current Q-function, using the reparameterization trick for gradient estimation. Target networks update via EMA (lines 15-16) to stabilize training.

The stochastic policy serves the same amortization purpose as in DDPG and TD3: it replaces the intractable $\arg\max$ operation with a fast network forward pass. SAC's entropy regularization produces exploration through the policy's inherent stochasticity rather than external noise. This makes SAC more robust to hyperparameters and eliminates the need to tune exploration schedules.

## Path Consistency Learning (PCL)

DDPG, TD3, and SAC all follow the same solution template from fitted Q-iteration: compute Bellman targets using the current Q-function, fit the Q-function to those targets, repeat. This is **successive approximation**, the function iteration approach $v_{k+1} = \Proj \Bellman v_k$ from the [projection methods chapter](projection.md).

Path Consistency Learning (PCL) {cite:p}`Nachum2017` solves the Bellman equation differently. Instead of iterating the operator, it directly minimizes a **residual**. This is the least-squares approach from projection methods: solve $\Residual(v) = 0$ by minimizing $\|\Residual(v)\|^2$. The method exploits special structure (smooth Bellman operators under deterministic dynamics) that conventional methods cannot leverage.

### The Path Consistency Property

Consider the entropy-regularized Q-function Bellman equation from the [smoothing chapter](smoothing.md). Under general stochastic dynamics, it involves an expectation over next states:

$$
q^*(s,a) = r(s,a) + \gamma \mathbb{E}_{s'}[v^*(s')]
$$

Suppose the dynamics are deterministic: $s' = f(s,a)$. The next state is uniquely determined, so the expectation disappears:

$$
q^*(s,a) = r(s,a) + \gamma v^*(f(s,a))
$$

The value function relates to Q-functions through the soft-max:

$$
v^*(s) = \alpha \log \int_{\mathcal{A}} \exp(q^*(s,a)/\alpha) da
$$

Contrast two cases: general policies versus the optimal Boltzmann policy.

**For general policies**, the value equals an expectation:

$$
v^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot|s)}[q^\pi(s,a) - \alpha \log \pi(a|s)]
$$ (eq:v-general-policy)

This is an average. For a single observed action $a$, we have:

$$
q^\pi(s,a) - \alpha\log\pi(a|s) = v^\pi(s) + \varepsilon(s,a)
$$

where $\varepsilon(s,a)$ is sampling error with $\mathbb{E}_{a \sim \pi}[\varepsilon(s,a)] = 0$. Individual actions give noisy estimates that fluctuate around the mean.

**For the optimal policy under entropy regularization**, the Boltzmann structure produces an exact pointwise identity. The optimal policy is:

$$
\pi^*(a|s) = \frac{\exp(q^*(s,a)/\alpha)}{\exp(v^*(s)/\alpha)}
$$

Taking logarithms and rearranging:

$$
v^*(s) = q^*(s,a) - \alpha \log \pi^*(a|s) \quad \text{for all } a
$$ (eq:v-q-exact-boltzmann)

This holds exactly for every action $a$, not just in expectation. There is no sampling error. The advantage $q^*(s,a) - v^*(s)$ is encoded in the log-probability: suboptimal actions have low $q^*(s,a)$ but also large $-\alpha\log\pi^*(a|s)$ (low probability means large negative log-probability), and these terms balance exactly to give $v^*(s)$.

Now take a trajectory segment $(s_0, a_0, s_1, a_1, \ldots, s_d)$ where each transition follows the deterministic dynamics $s_{t+1} = f(s_t, a_t)$. Start with $q^*(s_0, a_0) = r_0 + \gamma v^*(s_1)$ and use equation {eq}`eq:v-q-exact-boltzmann` to substitute $v^*(s_1) = q^*(s_1,a_1) - \alpha\log\pi^*(a_1|s_1)$ exactly:

$$
q^*(s_0, a_0) = r_0 + \gamma[q^*(s_1,a_1) - \alpha\log\pi^*(a_1|s_1)]
$$

Substitute $q^*(s_1,a_1) = r_1 + \gamma v^*(s_2)$:

$$
q^*(s_0, a_0) = r_0 + \gamma r_1 - \gamma\alpha\log\pi^*(a_1|s_1) + \gamma^2 v^*(s_2)
$$

Continue this telescoping for $d$ steps. Each substitution is exact:

$$
q^*(s_0, a_0) = \sum_{t=0}^{d-1} \gamma^t r_t - \alpha \sum_{t=1}^{d-1} \gamma^t \log\pi^*(a_t|s_t) + \gamma^d v^*(s_d)
$$

Apply equation {eq}`eq:v-q-exact-boltzmann` once more to get $v^*(s_0) = q^*(s_0,a_0) - \alpha\log\pi^*(a_0|s_0)$:

$$
v^*(s_0) = \sum_{t=0}^{d-1} \gamma^t [r_t - \alpha\log\pi^*(a_t|s_t)] + \gamma^d v^*(s_d)
$$ (eq:path-consistency-exact)

Rearranging gives the **path consistency residual**:

$$
R(s_0:s_d; \pi^*, v^*) = v^*(s_0) - \gamma^d v^*(s_d) - \sum_{t=0}^{d-1} \gamma^t[r_t - \alpha\log\pi^*(a_t|s_t)] = 0
$$ (eq:path-residual)

The telescoping produces an exact identity: $R = 0$ for every action sequence, not just in expectation. The behavior policy never appears because the constraint holds as a deterministic identity for any observed $(s_0, a_0, \ldots, s_d)$. This enables off-policy learning without importance sampling.

```{prf:remark} Contrasting General Policies and Optimal Boltzmann Policies
:class: dropdown

The distinction between equations {eq}`eq:v-general-policy` and {eq}`eq:v-q-exact-boltzmann` is subtle but crucial.

**For general policies** (equation {eq}`eq:v-general-policy`), the value is an average over actions sampled from the policy. Individual actions give noisy estimates: if we draw $a \sim \pi(\cdot|s)$, then $q^\pi(s,a) - \alpha\log\pi(a|s) = v^\pi(s) + \varepsilon$ where $\varepsilon$ is a zero-mean random variable. We need to average many samples to estimate $v^\pi(s)$ accurately. Multi-step telescoping would accumulate these sampling errors $\varepsilon_0, \varepsilon_1, \ldots, \varepsilon_{d-1}$, producing noisy residuals even at the true solution. Off-policy learning would require importance weights to correct for using actions from a different behavior policy.

**For the optimal entropy-regularized policy** (equation {eq}`eq:v-q-exact-boltzmann`), the Boltzmann structure collapses the expectation to a pointwise identity. The relationship $v^*(s) = q^*(s,a) - \alpha\log\pi^*(a|s)$ holds exactly for every action $a$, optimal or not. A suboptimal action has low $q^*(s,a)$ (low expected return) and low $\pi^*(a|s)$ (low probability), making $-\alpha\log\pi^*(a|s)$ large. These terms balance precisely to give $v^*(s)$. No sampling error exists. The telescoping is exact, producing a residual that equals zero for every action sequence, not just in expectation. Off-policy learning works because the constraint holds as a deterministic identity for any observed path.

This property is unique to soft-max operators. For hard-max, $v^*(s) = \max_a q^*(s,a)$ holds only when $a$ is optimal. Suboptimal actions satisfy $v^*(s) > q^*(s,a)$, an inequality that cannot be used to construct a residual.
```

### Structural Requirements: Deterministic Dynamics and Entropy Regularization

PCL's two structural requirements (deterministic dynamics and entropy regularization) are not arbitrary design choices. Each addresses a fundamental theoretical issue.

#### Deterministic Dynamics: Avoiding the Double Sampling Problem

Under stochastic dynamics, the Q-function Bellman equation has an expectation over next states:

$$
q^*(s,a) = r(s,a) + \gamma \mathbb{E}_{s' \sim p(\cdot|s,a)}[v^*(s')]
$$

The exact relationship {eq}`eq:v-q-exact-boltzmann` still holds, so we can write the path consistency constraint. But now consider what PCL minimizes: the **squared** residual $\mathbb{E}[R^2]$ where

$$
R = v_{\boldsymbol{\phi}}(s_0) - \gamma^d v_{\boldsymbol{\phi}}(s_d) - \sum_{t=0}^{d-1} \gamma^t[r_t - \alpha\log\pi_{\boldsymbol{\theta}}(a_t|s_t)]
$$

At the true optimum $(v^*, \pi^*)$, the constraint is $\mathbb{E}[R] = 0$, which implies $(\mathbb{E}[R])^2 = 0$. But PCL minimizes $\mathbb{E}[R^2]$, and by Jensen's inequality:

$$
\mathbb{E}[R^2] \geq (\mathbb{E}[R])^2
$$

with equality only when $R$ has zero variance. Under stochastic dynamics, even at optimality, individual trajectory residuals are random variables with mean zero but positive variance (due to transition noise). Minimizing $\mathbb{E}[R^2]$ to zero would require driving $\text{Var}(R) \to 0$, which is impossible and pushes the solution away from the true optimum.

This is Baird's **double sampling problem** {cite:p}`Baird1995`. To get an unbiased gradient of $(\mathbb{E}[R])^2$, we need:

$$
\nabla (\mathbb{E}[R])^2 = 2\mathbb{E}[R] \cdot \nabla \mathbb{E}[R] = 2\mathbb{E}[R] \cdot \mathbb{E}[\nabla R]
$$

This requires two independent samples of the next state from the same $(s,a)$ pair: one for estimating $\mathbb{E}[R]$ and one for $\mathbb{E}[\nabla R]$. With a simulator, this is possible. With real trajectories, it is not.

Under deterministic dynamics, $R$ is deterministic (no transition noise), so $\mathbb{E}[R^2] = (\mathbb{E}[R])^2$ and Jensen's inequality holds with equality. Minimizing the squared residual is equivalent to solving $\mathbb{E}[R] = 0$.

#### Entropy Regularization: Enabling All-Action Consistency

Attempt the same path consistency derivation with the hard-max Bellman operator. Under deterministic dynamics, the Q-function satisfies:

$$
q^*(s,a) = r(s,a) + \gamma v^*(f(s,a))
$$

where $v^*(s) = \max_{a'} q^*(s,a')$ and the optimal policy is $\pi^*(s) = \arg\max_a q^*(s,a)$ (deterministic).

Now try to relate $v^*(s)$ to an arbitrary observed action $a$. For the optimal action $a^* \in \arg\max_{a'} q^*(s,a')$, we have:

$$
v^*(s) = q^*(s,a^*)
$$

But for a suboptimal action $a \ne a^*$:

$$
v^*(s) = \max_{a'} q^*(s,a') > q^*(s,a)
$$

This is an inequality, not an equation. There is no formula expressing $v^*(s)$ in terms of $q^*(s,a)$ for suboptimal actions.

Attempt the multi-step telescoping. Start with $q^*(s_0, a_0) = r_0 + \gamma v^*(s_1)$. To continue, we need to express $v^*(s_1)$ using the observed action $a_1$. But we only have:

$$
v^*(s_1) \geq q^*(s_1, a_1)
$$

with equality only if $a_1$ happens to be optimal at $s_1$. We cannot substitute this into the Q-function equation to get an exact telescoping. The derivation breaks at the first step.

Compare this to the soft-max case. The Boltzmann structure gives equation {eq}`eq:v-q-exact-boltzmann`: $v^*(s) = q^*(s,a) - \alpha\log\pi^*(a|s)$ for all actions $a$. The log-probability term compensates exactly for suboptimality: low-probability actions have large $-\alpha\log\pi^*(a|s)$, which adds to the low $q^*(s,a)$ to recover $v^*(s)$. This enables exact substitution at every step:

$$
v^*(s_1) = q^*(s_1, a_1) - \alpha\log\pi^*(a_1|s_1) \quad \text{(exact for any } a_1\text{)}
$$

The telescoping proceeds without inequalities or restrictions on which actions were chosen. Multi-step hard-max Q-learning lacks theoretical justification for off-policy data because when we observe a trajectory with suboptimal actions, we cannot write an exact path consistency constraint.

Both requirements are structural:

| **Requirement** | **Addresses** |
|:----------------|:--------------|
| Deterministic dynamics | Double sampling bias: ensures $\mathbb{E}[R^2] = (\mathbb{E}[R])^2$ |
| Entropy regularization | All-action consistency (equation {eq}`eq:v-q-exact-boltzmann`) |

Without deterministic dynamics, residual minimization is biased. Without entropy regularization, the constraint holds only for optimal actions.

### The Learning Objective

Equation {eq}`eq:path-residual` provides a constraint that the optimal $(v^*, \pi^*)$ must satisfy: the residual equals zero for every observed path. For parametric approximations $(v_{\boldsymbol{\phi}}, \pi_{\boldsymbol{\theta}})$ that are not yet optimal, the residual is nonzero:

$$
R(s_0:s_d; \boldsymbol{\theta}, \boldsymbol{\phi}) = v_{\boldsymbol{\phi}}(s_0) - \gamma^d v_{\boldsymbol{\phi}}(s_d) - \sum_{t=0}^{d-1} \gamma^t[r_t - \alpha \log \pi_{\boldsymbol{\theta}}(a_t|s_t)]
$$

PCL minimizes the squared residual over observed path segments:

$$
\min_{\boldsymbol{\theta}, \boldsymbol{\phi}} \sum_{\text{segments}} \frac{1}{2} R(s_i:s_{i+d}; \boldsymbol{\theta}, \boldsymbol{\phi})^2
$$

This is the least-squares residual approach from the [projection methods chapter](projection.md). SAC computes targets $y_i$ and fits to them (successive approximation). PCL directly minimizes the residual without computing targets or performing a separate fitting step.

Gradient descent gives:

$$
\begin{align}
\boldsymbol{\theta}_{k+1} &= \boldsymbol{\theta}_k + \eta_\pi \sum_i R_i \cdot \alpha \sum_{t=0}^{d-1} \gamma^t \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}_k}(a_{i+t}|s_{i+t}) \\
\boldsymbol{\phi}_{k+1} &= \boldsymbol{\phi}_k - \eta_v \sum_i R_i \left[\nabla_{\boldsymbol{\phi}} v_{\boldsymbol{\phi}_k}(s_i) - \gamma^d \nabla_{\boldsymbol{\phi}} v_{\boldsymbol{\phi}_k}(s_{i+d})\right]
\end{align}
$$

where $R_i = R(s_i:s_{i+d}; \boldsymbol{\theta}_k, \boldsymbol{\phi}_k)$. Large residuals drive larger updates.

```{prf:algorithm} Path Consistency Learning (PCL)
:label: pcl

**Input:** MDP with deterministic dynamics $s_{t+1} = f(s_t, a_t)$, policy $\pi_{\boldsymbol{\theta}}$, value function $v_{\boldsymbol{\phi}}$, entropy weight $\alpha$, path length $d$, learning rates $\eta_\pi, \eta_v$, replay buffer capacity $B$

**Output:** Policy parameters $\boldsymbol{\theta}$, value parameters $\boldsymbol{\phi}$

1. Initialize $\boldsymbol{\theta}_0$, $\boldsymbol{\phi}_0$
2. Initialize replay buffer $\mathcal{R}$ with capacity $B$
3. $k \leftarrow 0$
4. **while** training **do**
    1. Sample trajectory $\tau = (s_0, a_0, r_0, \ldots, s_T)$ from $\pi_{\boldsymbol{\theta}_k}$ and store in $\mathcal{R}$
    2. Sample trajectory $\tau'$ from $\mathcal{R}$
    3. **for** each $d$-step segment in $\tau'$ **do**
        1. Compute residual: $R_i \leftarrow v_{\boldsymbol{\phi}_k}(s_i) - \gamma^d v_{\boldsymbol{\phi}_k}(s_{i+d}) - \sum_{t=0}^{d-1} \gamma^t[r_{i+t} - \alpha \log \pi_{\boldsymbol{\theta}_k}(a_{i+t}|s_{i+t})]$
        2. Update policy: $\boldsymbol{\theta}_{k+1} \leftarrow \boldsymbol{\theta}_k + \eta_\pi \alpha R_i \sum_{t=0}^{d-1} \gamma^t \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}_k}(a_{i+t}|s_{i+t})$
        3. Update value: $\boldsymbol{\phi}_{k+1} \leftarrow \boldsymbol{\phi}_k - \eta_v R_i\left[\nabla_{\boldsymbol{\phi}} v_{\boldsymbol{\phi}_k}(s_i) - \gamma^d \nabla_{\boldsymbol{\phi}} v_{\boldsymbol{\phi}_k}(s_{i+d})\right]$
    4. Remove oldest trajectories if $|\mathcal{R}| > B$
    5. $k \leftarrow k + 1$
5. **return** $\boldsymbol{\theta}_k$, $\boldsymbol{\phi}_k$
```

The algorithm collects trajectories from the current policy and stores them in a replay buffer. At each iteration, it samples a trajectory (possibly old) and performs gradient descent on the path residual for all $d$-step segments. The replay buffer enables off-policy learning: trajectories from old policies, expert demonstrations, or exploratory behavior all provide valid training signals.

### Unified Parameterization: Single Q-Network

Algorithm {prf:ref}`pcl` uses separate networks for policy and value. But we can use a single Q-network $q_{\boldsymbol{\theta}}(s,a)$ and derive both:

$$
v_{\boldsymbol{\theta}}(s) = \alpha \log \sum_{a} \exp(q_{\boldsymbol{\theta}}(s,a)/\alpha), \qquad \pi_{\boldsymbol{\theta}}(a|s) = \frac{\exp(q_{\boldsymbol{\theta}}(s,a)/\alpha)}{\sum_{a'} \exp(q_{\boldsymbol{\theta}}(s,a')/\alpha)}
$$

The path residual becomes:

$$
R(s_i:s_{i+d}; \boldsymbol{\theta}) = v_{\boldsymbol{\theta}}(s_i) - \gamma^d v_{\boldsymbol{\theta}}(s_{i+d}) - \sum_{t=0}^{d-1} \gamma^t[r_{i+t} - \alpha \log \pi_{\boldsymbol{\theta}}(a_{i+t}|s_{i+t})]
$$

and the gradient combines both value and policy contributions through the same parameters. This unified architecture eliminates the actor-critic separation: one Q-network serves both roles.

### Connection to Existing Methods

**Single-step case ($d=1$)**: The path residual becomes $R(s:s'; \boldsymbol{\theta}, \boldsymbol{\phi}) = v_{\boldsymbol{\phi}}(s) - \gamma v_{\boldsymbol{\phi}}(s') - r + \alpha\log\pi_{\boldsymbol{\theta}}(a|s)$. For unified parameterization where $v_{\boldsymbol{\theta}}(s) = q_{\boldsymbol{\theta}}(s,a) - \alpha\log\pi_{\boldsymbol{\theta}}(a|s)$ exactly, this becomes $R = q_{\boldsymbol{\theta}}(s,a) - r - \gamma v_{\boldsymbol{\theta}}(s')$, the soft Bellman residual. Minimizing $\sum_i R_i^2$ is equivalent to soft Q-learning, though SAC solves this via successive approximation (compute targets, fit) rather than direct residual minimization.

**No entropy ($\alpha \to 0$)**: The residual becomes $R = v(s_i) - \gamma^d v(s_{i+d}) - \sum_t \gamma^t r_t$, the negative $d$-step advantage. But unlike A2C/A3C where $v$ tracks the current policy's value, PCL's value converges to $v^*$ because the residual couples policy and value through the optimality condition.

**Multi-step with hard-max**: No analog exists. The hard-max Bellman operator $\max_a q(s,a)$ does not have an exact pointwise relationship like equation {eq}`eq:v-q-exact-boltzmann`. Multi-step telescoping would accumulate errors from the max operator, making the constraint valid only in expectation under the optimal policy. The soft-max structure enables exact off-policy path consistency.

### PCL vs SAC: Residual Minimization vs Successive Approximation

Both methods solve entropy-regularized MDPs but use fundamentally different solution strategies:

| **Aspect** | **SAC** | **PCL** |
|:-----------|:--------|:--------|
| **Solution method** | Successive approximation: compute targets $y_i$, fit $q$ to targets | Residual minimization: minimize $\sum_i R_i^2$ directly |
| **Update structure** | Target computation + regression step | Single gradient step on squared residual |
| **Target networks** | Required (mark outer-iteration boundaries) | None (residual constraint, not target fitting) |
| **Temporal horizon** | Single-step TD: $y = r + \gamma V(s')$ | Multi-step paths: accumulate over $d$ steps |
| **Off-policy handling** | Replay buffer with single-sample bias | No importance sampling (works for any trajectory) |
| **Dynamics requirement** | General stochastic transitions | **Deterministic** transitions $s' = f(s,a)$ |
| **Architecture** | Twin Q-networks + policy network | Single Q-network (unified parameterization) |

PCL requires deterministic dynamics. It gains multi-step telescoping and off-policy learning without importance weights, but only for deterministic systems (robotic manipulation, many control tasks). SAC works for general stochastic MDPs.

### PCL as Amortization

PCL amortizes at a different level than DDPG/TD3/SAC. Those methods amortize the action maximization: learn a policy network that outputs $\arg\max_a q(s,a)$ directly. PCL amortizes the solution of the Bellman equation itself. Instead of repeatedly applying the Bellman operator (which requires $\int_{\mathcal{A}} \exp(q/\alpha) da$ at every iteration), PCL samples path segments and minimizes their residual. The computational cost of verifying optimality across all states and path lengths is distributed across training through sampled gradient updates.

# Summary

This chapter addressed the computational barrier that arises when extending value-based fitted methods to continuous action spaces. The core issue is tractability: computing $\max_{a \in \mathbb{R}^m} q(s,a;\boldsymbol{\theta})$ at each Bellman target evaluation requires solving a nonlinear optimization problem. For replay buffers containing millions of transitions, repeatedly solving these optimization problems becomes prohibitive.

The solution we developed is **amortization**: invest computational effort during training to learn a policy network that replaces runtime optimization with fast forward passes. This strategy keeps us firmly within the dynamic programming framework. We still compute Bellman operators and maintain Q-functions. The amortization makes these operations tractable by replacing explicit $\arg\max$ operations with learned policy networks.

Most methods (NFQCA, DDPG, TD3, SAC) follow the **successive approximation** paradigm: compute Bellman targets, fit to targets, repeat. PCL takes a different approach, directly minimizing a **residual** (the path consistency residual) rather than iterating the Bellman operator. This aligns PCL with least-squares residual methods from the [projection methods chapter](projection.md), rather than function iteration.

All methods share the core amortization idea but differ along several dimensions:

**Solution methodology.** NFQCA, DDPG, TD3, and SAC use **successive approximation** (function iteration): compute Bellman targets, fit to targets, repeat. This is the $v_{k+1} = \Proj \Bellman v_k$ paradigm from the [projection methods chapter](projection.md). PCL uses **residual minimization**: directly minimize the squared path residual $\sum_i R_i^2$ via gradient descent. This is the least-squares approach where we solve $\Residual(v) = 0$ by minimizing $\|\Residual(v)\|^2$ rather than iterating the operator.

**Policy class**: Deterministic policies (NFQCA, DDPG, TD3) output a single action $\pi_{\boldsymbol{w}}(s)$ and approximate $\arg\max_a q(s,a;\boldsymbol{\theta})$ by maximizing $q(s,\pi_{\boldsymbol{w}}(s);\boldsymbol{\theta})$. This requires external exploration noise during training. Stochastic policies (SAC, PCL) output a distribution $\pi_{\boldsymbol{w}}(a|s)$ and approximate the Boltzmann distribution under entropy regularization. Exploration comes from sampling the stochastic policy.

**Temporal structure**: Single-step methods (NFQCA, DDPG, TD3, SAC) use one-step Bellman backups with targets $r + \gamma V(s')$. SAC estimates $V(s')$ via single-sample Monte Carlo. PCL exploits deterministic dynamics to chain the Bellman equation over $d$ steps, using observed action sequences and accumulating rewards along entire path segments.

**Target networks**: Methods based on successive approximation (NFQCA, DDPG, TD3, SAC) use target networks that mark outer-iteration boundaries in the flattened FQI structure. PCL has no target networks because it minimizes a residual rather than fitting to computed targets. The twin Q-network trick (TD3, SAC) uses $\min(q^1, q^2)$ to mitigate overestimation; PCL avoids this issue through the residual minimization structure.

All methods remain fundamentally value-based: they maintain Q-functions, compute approximate Bellman operators, and derive policies from learned value estimates. This connection to dynamic programming provides theoretical grounding (we know these methods implement approximate successive approximation of the Bellman equation) but also imposes structure. The policy must track the Q-function's implied greedy policy (in DDPG/TD3) or Boltzmann distribution (in SAC). When the Q-function is inaccurate, which is inevitable with function approximation, the policy inherits these errors.

The next chapter takes a different perspective. Rather than extending DP-based value methods to continuous actions through amortization, we parameterize the policy directly and optimize it via gradient ascent on expected return. This shifts the fundamental question from "how do we compute $\max_a q(s,a)$ efficiently?" to "how do we estimate $\nabla_{\boldsymbol{\theta}} \mathbb{E}_{\tau \sim p_{\boldsymbol{\theta}}}[R(\tau)]$ accurately?" The resulting policy gradient methods are DP-agnostic: they work without Bellman equations, Q-functions, or value estimates. This removes the scaffolding of dynamic programming while introducing new challenges in gradient estimation and variance reduction.

