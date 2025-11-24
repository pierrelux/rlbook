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

NFQCA {cite}`Hafner2011` extends the NFQI template from the [previous chapter](fqi.md) to handle continuous action spaces by replacing the $\max_{a'} q(s',a'; \boldsymbol{\theta})$ operator in the Bellman target with a parameterized policy $\pi_{\boldsymbol{w}}(s')$. This transforms fitted Q-iteration into an actor-critic method: the critic $q(s,a; \boldsymbol{\theta})$ evaluates state-action pairs via the standard regression step, while the actor $\pi_{\boldsymbol{w}}(s)$ provides actions by directly maximizing the learned Q-function.

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

The algorithm structure mirrors NFQI (Algorithm {prf:ref}`fitted-q-iteration-batch` in the [FQI chapter](fqi.md)) with two key extensions. First, target computation (line 7-8) replaces the discrete max with a policy network call $\pi_{\boldsymbol{w}_n}(s')$, making the Bellman operator tractable for continuous actions. Second, after fitting the Q-function (line 11), we add a policy improvement step (line 13) that updates $\boldsymbol{w}$ to maximize the Q-function evaluated at policy-generated actions over states in the dataset.

Both `fit` operations use gradient descent with warm starting, consistent with the NFQI template. The Q-function minimizes squared Bellman error using targets computed with the current policy. The policy maximizes the Q-function via gradient ascent on the composition $q(s, \pi_{\boldsymbol{w}}(s); \boldsymbol{\theta}_{n+1})$, which is differentiable end-to-end when both networks are differentiable. The gradient with respect to $\boldsymbol{w}$ is:

$$
\nabla_{\boldsymbol{w}} q(s, \pi_{\boldsymbol{w}}(s); \boldsymbol{\theta}) = \nabla_a q(s, a; \boldsymbol{\theta})\Big|_{a=\pi_{\boldsymbol{w}}(s)} \cdot \nabla_{\boldsymbol{w}} \pi_{\boldsymbol{w}}(s)
$$

computed via the chain rule (backpropagation through the actor into the critic). Modern automatic differentiation libraries handle this composition automatically.

## Euler Equation Methods: Approximating Policies Instead of Values

The weighted residual framework applies to any functional equation arising from an MDP. So far we have applied it to the Bellman equation in order to approximate the value function. An alternative is to approximate the **optimal policy** directly by enforcing the **Euler equations**, that is, the first-order optimality conditions of the dynamic optimization problem {cite}`Judd1992,Rust1996,ndp`.

### From Bellman to Euler equations

Consider a discounted control problem with continuous states and actions. Let the state space be $\mathcal{S} \subset \mathbb{R}^d$, the action space be $\mathcal{A} \subset \mathbb{R}^m$, and the dynamics be deterministic,

$$
s' = f(s,a),
$$

with $f$ continuously differentiable in both arguments. The reward function $r(s,a)$ is also continuously differentiable, and the discount factor satisfies $0 < \gamma < 1$.

The optimal value function $v^*$ solves the Bellman equation

$$
v^*(s)
=
\max_{a \in \mathcal{A}}
\left\{ r(s,a) + \gamma\, v^*\big(f(s,a)\big) \right\}.
$$

Assume that for each state $s$, the maximizer is unique, and denote the corresponding optimal policy by $\pi^*(s)$:

$$
\pi^*(s)
\in
\arg\max_{a \in \mathcal{A}}
\left\{ r(s,a) + \gamma\, v^*\big(f(s,a)\big) \right\}.
$$

For a fixed $s$, define

$$
g_s(a) := r(s,a) + \gamma\, v^*\big(f(s,a)\big).
$$

If $\pi^*(s)$ is an unconstrained maximizer of $g_s$, then the first-order condition is

$$
0
=
\frac{\partial g_s}{\partial a}(a)\Big|_{a=\pi^*(s)}
=
\frac{\partial r}{\partial a}\big(s,\pi^*(s)\big)
+
\gamma\,
\frac{\partial v^*}{\partial s}\big(f(s,\pi^*(s))\big)\,
\frac{\partial f}{\partial a}\big(s,\pi^*(s)\big),
$$

which we write as the **Euler equation**

$$
\frac{\partial r}{\partial a}\big(s,\pi^*(s)\big)
+
\gamma\, \frac{\partial v^*}{\partial s}\big(f(s,\pi^*(s))\big)\,
\frac{\partial f}{\partial a}\big(s,\pi^*(s)\big)
=
0,
\quad \forall s \in \mathcal{S}.
$$ (Euler-raw)

This "raw" Euler equation still involves the unknown derivative $\partial v^*/\partial s$. In a general MDP, any method based directly on the raw Euler equation would have to solve jointly for both the policy and the value-function derivatives.

### Euler-class structure

Many control problems have additional structure that allows us to eliminate $\partial v^*/\partial s$ entirely from the Euler equation. Rust formalizes such problems as an **Euler class** of continuous decision processes {cite}`Rust1996`. This is not simply "any problem where an Euler equation exists" (which would include almost all smooth MDPs), but rather a narrow class with special structural properties.

The key feature is that the state can be decomposed into two parts:

$$
s = (x, \xi),
$$

where

- $x \in \mathbb{R}^{d_x}$ is a **controlled component** that the agent directly influences through actions (e.g., inventory level, battery charge, reservoir water level),
- $\xi \in \mathbb{R}^{d_\xi}$ is an **exogenous component** that evolves independently of actions (e.g., demand, weather, electricity prices).

The transition law has the special factorized form

$$
p(x', \xi' \mid x, \xi, a)
=
\mathbf{1}\{x' = f(x, a, \xi, \xi')\} \cdot q(\xi' \mid \xi),
$$

where $q$ governs the exogenous component $\xi'$ (independent of the action) and $f$ is a continuously differentiable deterministic function giving the next controlled state. The reward function $r(x, \xi, a)$ is continuously differentiable and typically concave in $(x,a)$ for each fixed $\xi$.

The key **Euler-class condition** is that there exists a matrix-valued function $h(x, a, \xi)$ such that

$$
\frac{\partial f(x, a, \xi, \xi')}{\partial x}
=
\frac{\partial f(x, a, \xi, \xi')}{\partial a} \cdot h(x, a, \xi)
\quad \text{for all } (x, a, \xi, \xi').
$$ (EC)

This is a **rank-one type condition**: every column of $\partial f/\partial x$ must lie in the column space of $\partial f/\partial a$. Geometrically, this says that the way the current controlled state $x$ affects the next controlled state $x'$ is "spanned" by the directions in which the action $a$ moves $x'$. In scalar problems, it simply means $\partial f/\partial x$ and $\partial f/\partial a$ are proportional.

#### Why this condition enables value elimination

Under these assumptions, one can derive an **envelope formula** expressing $\partial v^*/\partial x$ purely in terms of $r$, $h$, and the optimal policy $\pi^*(x, \xi)$:

$$
\frac{\partial v^*(x, \xi)}{\partial x}
=
\frac{\partial r(x, \xi, \pi^*(x, \xi))}{\partial x}
+
\frac{\partial r(x, \xi, \pi^*(x, \xi))}{\partial a} \cdot h(x, \pi^*(x, \xi), \xi).
$$

This formula expresses the value gradient entirely in terms of observable primitives (the reward function $r$ and the function $h$ from the dynamics) and the policy $\pi^*$—with no explicit dependence on $v^*$ itself. Substituting this envelope formula into the raw first-order condition eliminates $\partial v^*/\partial x$ entirely, yielding an **Euler equation that is a closed functional equation in the policy $\pi^*$ alone**.

The key insight is that the special derivative structure of the dynamics allows us to trace how changes in the controlled state affect future value purely through their effect on the immediate reward and transition structure, without needing to track the value function separately.

Concretely, the Bellman equation becomes

$$
v^*(x, \xi)
=
\max_{a \in \mathcal{A}(x, \xi)}
\left[
r(x, \xi, a)
+
\gamma \int v^*\big(f(x, a, \xi, \xi'), \xi'\big)\, q(d\xi' \mid \xi)
\right],
$$

and the optimal policy $\pi^*(x, \xi)$ satisfies a stochastic Euler equation of the form

$$
\mathcal{E}(\pi)(x, \xi) = 0,
\quad \forall (x, \xi),
$$ (Euler)

where $\mathcal{E}(\pi)(x, \xi)$ depends only on current and next-period $(x, \xi, a)$, $(x', \xi', a')$ and the primitives $r, f, h, q$, but not directly on $v^*$ or its derivatives.

### Examples of Euler-class dynamics

The Euler-class condition may look abstract, but it is easy to verify in many control problems. In particular, **affine dynamics** in $(x,a)$ automatically satisfy the condition, since the derivatives are constant.

More generally, any problem with:
- A low-dimensional controlled state whose dynamics are affine or simple functions of the current state and action,
- An exogenous component that evolves independently of actions, and
- Concave stage reward,

is a candidate for the Euler-class framework. This includes inventory, energy storage, reservoir, and thermal control problems.

We illustrate with simple scalar examples, where $x, a \in \mathbb{R}$ and $\xi$ is an exogenous component.

#### Inventory / production planning

Let

$$
x' = f(x, a, \xi, \xi')
= \alpha x + a - D(\xi'),
$$

where $x$ is the inventory level, $a$ is the order/production quantity, $0 < \alpha \le 1$ models carry-over/spoilage, and $D(\xi')$ is demand (a function of the exogenous component).

Then

$$
\frac{\partial f}{\partial x} = \alpha,
\qquad
\frac{\partial f}{\partial a} = 1.
$$

The Euler-class condition holds with

$$
h(x, a, \xi) = \alpha,
$$

since

$$
\frac{\partial f}{\partial x}
=
\frac{\partial f}{\partial a} \cdot h
\quad \Leftrightarrow \quad
\alpha = 1 \cdot \alpha.
$$

#### Energy storage / battery arbitrage

Consider a simple (smoothed) storage model

$$
x'
=
f(x, a, \xi, \xi')
=
\eta_{\mathrm{ret}} x + \eta_{\mathrm{ch}} a,
$$

where $x$ is the state of charge, $a$ is the charging power, and $\eta_{\mathrm{ret}}, \eta_{\mathrm{ch}}$ are retention and charging efficiency parameters.

Then

$$
\frac{\partial f}{\partial x} = \eta_{\mathrm{ret}},
\qquad
\frac{\partial f}{\partial a} = \eta_{\mathrm{ch}},
$$

so the Euler-class condition holds with

$$
h(x, a, \xi) = \frac{\eta_{\mathrm{ret}}}{\eta_{\mathrm{ch}}}.
$$

#### Reservoir / irrigation control

Let $x$ be the water level in a reservoir, $a$ the release, and $I(\xi')$ the inflow. A simple water-balance model is

$$
x'
=
f(x, a, \xi, \xi')
=
\alpha x + I(\xi') - a,
$$

with $0 < \alpha \le 1$ capturing evaporation/seepage losses.

Then

$$
\frac{\partial f}{\partial x} = \alpha,
\qquad
\frac{\partial f}{\partial a} = -1,
$$

so the Euler-class condition holds with

$$
h(x, a, \xi) = -\alpha,
$$

since

$$
\frac{\partial f}{\partial x}
=
\frac{\partial f}{\partial a} \cdot h
\quad \Leftrightarrow \quad
\alpha = (-1)\cdot(-\alpha).
$$

#### Coarse building thermal mass model

In a very simple RC model for a building zone, one might write

$$
x'
=
f(x, a, \xi, \xi')
=
\alpha(\xi) x + \beta(\xi) a + \delta(\xi),
$$

where $x$ is an internal energy or temperature proxy, $a$ is the HVAC control input, and $\xi$ captures exogenous conditions (outside temperature, occupancy mode).

Then

$$
\frac{\partial f}{\partial x} = \alpha(\xi),
\qquad
\frac{\partial f}{\partial a} = \beta(\xi),
$$

and the Euler-class condition holds with

$$
h(x, a, \xi) = \frac{\alpha(\xi)}{\beta(\xi)},
$$

provided $\beta(\xi) \neq 0$.

#### What these examples reveal

In all these examples, the dynamics are **affine** in $(x,a)$:

$$
x' = \alpha(\xi, \xi') x + \beta(\xi, \xi') a + \delta(\xi, \xi'),
$$

where $\alpha, \beta, \delta$ may depend on the exogenous components but not on $(x,a)$. For such dynamics:

$$
\frac{\partial f}{\partial x} = \alpha(\xi, \xi'), 
\qquad 
\frac{\partial f}{\partial a} = \beta(\xi, \xi'),
$$

so the Euler-class condition holds with $h(x, a, \xi) = \alpha(\xi, \xi')/\beta(\xi, \xi')$ (assuming $\beta \neq 0$).

More generally, even when dynamics are not perfectly affine, if they take the form

$$
x' = g(\xi, \xi') \cdot \big(x + \phi(a)\big),
$$

we get

$$
\frac{\partial f}{\partial x} = g(\xi, \xi'), 
\qquad 
\frac{\partial f}{\partial a} = g(\xi, \xi') \cdot \phi'(a),
$$

so $h(x, a, \xi) = 1/\phi'(a)$.

The key insight: **affine or near-affine dynamics make the Euler-class condition automatic**. This is why Euler-type methods apply to many control problems: inventory, storage, reservoir, and thermal control—anywhere the controlled state dynamics can be approximated as affine or near-affine.

### The Euler residual operator

For Euler-class problems, the Euler equation can be written as

$$
\mathcal{E}(\pi)(x, \xi) = 0,
\quad \forall (x, \xi).
$$

Define the **Euler residual operator** $\mathcal{R}_E$ by

$$
\mathcal{R}_E(\pi)(x, \xi)
:=
\mathcal{E}(\pi)(x, \xi),
$$

that is, $\mathcal{R}_E(\pi)$ is the left-hand side of the Euler equation evaluated at the policy $\pi$. The optimal policy $\pi^*$ satisfies

$$
\mathcal{R}_E(\pi^*)(x, \xi) = 0
\quad \text{for all } (x, \xi).
$$

In practice, we work with a parameterized policy $\pi_\theta(x, \xi)$. The goal is to choose $\theta$ so that $\mathcal{R}_E(\pi_\theta)$ is "small" in some sense. For example, we can enforce the Euler equation at a finite set of collocation points.

### The nonlinear system to solve

Let $(x_1, \xi_1), \dots, (x_N, \xi_N)$ be a set of collocation points in the state space. Enforcing the Euler equation at these points gives the system

$$
\mathcal{R}_E(\pi_\theta)(x_k, \xi_k) = 0,
\quad k = 1,\dots,N.
$$

Equivalently, define the vector-valued function

$$
G(\theta)
:=
\begin{bmatrix}
\mathcal{R}_E(\pi_\theta)(x_1, \xi_1) \\
\vdots \\
\mathcal{R}_E(\pi_\theta)(x_N, \xi_N)
\end{bmatrix},
$$

so that the **Euler discretization** is the finite-dimensional nonlinear system

$$
G(\theta) = 0.
$$

In a Galerkin or least-squares variant, one chooses test functions $\{\varphi_i\}_{i=1}^M$ and enforces orthogonality conditions

$$
\int \varphi_i(x, \xi) \mathcal{R}_E(\pi_\theta)(x, \xi) \mu(dx, d\xi) = 0,
\quad i = 1,\dots,M,
$$

for a chosen weighting measure $\mu$. These conditions can again be collected into a vector equation

$$
G(\theta) = 0,
$$

where each component of $G$ corresponds to a weighted Euler residual.

The mapping $G$ is generally nonlinear in $\theta$, so one solves $G(\theta) = 0$ using nonlinear equation solvers (Newton-type methods, fixed-point iterations, or other root-finding schemes). Unlike the Bellman operator, the Euler operator is not a contraction in a natural norm, so generic convergence guarantees are weaker and depend on the specific problem.

```{admonition} Why the Euler class is special
:class: note

It's important to understand that the **Euler class** is not "all problems where you can write first-order conditions." Almost any smooth MDP with continuous actions has first-order optimality conditions.

What makes the Euler class special is the structural condition that allows **complete elimination** of the value function from the system. In a general MDP, the raw first-order condition $\frac{\partial r}{\partial a} + \gamma \frac{\partial v^*}{\partial s} \frac{\partial f}{\partial a} = 0$ still involves $\partial v^*/\partial s$, so you'd need to approximate both the policy and the value gradient jointly.

The Euler-class condition—that $\partial f/\partial x = (\partial f/\partial a) h$—is what enables the envelope formula that expresses $\partial v^*/\partial x$ purely in terms of primitives and the policy. This transforms a coupled system (policy + value) into a **closed functional equation in the policy alone**.

This allows Euler equation methods to define an operator on policies and apply projection methods without carrying around a separate approximation to $v^*$ or its derivatives—a significant computational advantage when the policy space has lower dimension than the value function space.
```

### Summary

- In a generic continuous-action MDP, the raw first-order condition involves $\partial v^*/\partial s$, so one would have to solve jointly for the policy and the value-function derivatives.
- **Euler-class** problems have additional structure on the dynamics that allows the derivative of the value function to be eliminated via an envelope formula. The resulting equation is a closed functional equation in the policy alone.
- Many control problems fit this structure: inventory models, energy storage, reservoir control, and thermal control models—any problem where the controlled state dynamics are affine or near-affine.
- Euler-equation methods for such problems apply weighted residual techniques directly to the Euler residual operator $\mathcal{R}_E(\pi)$, leading to a finite-dimensional nonlinear system $G(\theta)=0$ in the policy parameters, rather than a Bellman fixed point in $v$.

## Deep Deterministic Policy Gradient (DDPG)

We now extend NFQCA to the online setting with evolving replay buffers, mirroring how DQN extended NFQI in the [FQI chapter](fqi.md). Just as DQN allowed $\mathcal{B}_t$ and $\hat{P}_{\mathcal{B}_t}$ to evolve during learning instead of using a fixed offline dataset, DDPG {cite}`lillicrap2015continuous` collects new transitions during training and stores them in a circular replay buffer.

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

**Input**: MDP $(S, A, P, R, \gamma)$, Q-network $q(s,a; \boldsymbol{\theta})$, policy network $\pi_{\boldsymbol{w}}(s)$, learning rates $\alpha_q, \alpha_\pi$, target update frequency $K$, replay buffer capacity $B$, mini-batch size $b$, exploration noise $\eta$

**Output**: Q-function parameters $\boldsymbol{\theta}$, policy parameters $\boldsymbol{w}$

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
    7. For each sampled transition $(s_i,a_i,r_i,s_i')$:
        1. $y_i \leftarrow r_i + \gamma q(s'_i, \pi_{\boldsymbol{w}_{\text{target}}}(s'_i); \boldsymbol{\theta}_{\text{target}})$ (actor target selects, critic target evaluates)
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

DDPG inherits the overestimation bias from DQN's use of the max operator in Bellman targets. TD3 {cite}`fujimoto2018addressing` addresses this through three modifications to the DDPG template, following similar principles to Double DQN but adapted for continuous actions and taking a more conservative approach.

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

**Input**: MDP $(S, A, P, R, \gamma)$, twin Q-networks $q^A(s,a; \boldsymbol{\theta}^A)$, $q^B(s,a; \boldsymbol{\theta}^B)$, policy network $\pi_{\boldsymbol{w}}(s)$, learning rates $\alpha_q, \alpha_\pi$, replay buffer capacity $B$, mini-batch size $b$, policy delay $d$, EMA rate $\tau$, target noise $\sigma$, noise clip $c$, exploration noise $\sigma_{\text{explore}}$

**Output**: Twin Q-function parameters $\boldsymbol{\theta}^A, \boldsymbol{\theta}^B$, policy parameters $\boldsymbol{w}$

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
    8. For each sampled transition $(s_i,a_i,r_i,s_i')$:
        1. $\tilde{a}_i \leftarrow \pi_{\boldsymbol{w}_{\text{target}}}(s'_i) + \text{clip}(\varepsilon_i, -c, c)$, $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ (target policy smoothing)
        2. $y_i \leftarrow r_i + \gamma \color{blue}{\min\big(q^A(s'_i, \tilde{a}_i; \boldsymbol{\theta}^A_{\text{target}}), q^B(s'_i, \tilde{a}_i; \boldsymbol{\theta}^B_{\text{target}})\big)}$ (minimum of twin targets)
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

Adapting NFQCA to the smooth Bellman optimality equations leads to the soft actor-critic algorithm {cite}`haarnoja2018soft`. To understand this connection, we first examine how the smooth Bellman equations follow from entropy regularization.

## Entropy-Regularized Bellman Optimality

Consider the entropy-regularized MDP objective where we seek a policy that maximizes both cumulative reward and policy entropy. The smooth Bellman operator $\mathrm{L}_\alpha$ takes the form:

$$
(\mathrm{L}_\alpha v)(s) = \max_{d(\cdot|s) \in \Delta(\mathcal{A})}\left\{\sum_a d(a|s)[r(s,a) + \gamma \mathbb{E}_{s'}[v(s')]] + \alpha \mathcal{H}(d(\cdot|s))\right\}
$$

where $\mathcal{H}(d(\cdot|s)) = -\sum_a d(a|s)\log d(a|s)$ represents the entropy of the decision rule at state $s$, $\alpha > 0$ is the entropy regularization weight, and $\Delta(\mathcal{A})$ denotes the probability simplex over actions.

To find the optimal decision rule, we solve a constrained optimization problem. For a fixed state $s$, we seek to maximize the objective function

$$
J(d) = \sum_a d(a|s)\left[r(s,a) + \gamma \mathbb{E}_{s'}[v(s')]\right] + \alpha \sum_a d(a|s)\log d(a|s)
$$

subject to the normalization constraint $\sum_a d(a|s) = 1$. 

We form the Lagrangian by introducing the multiplier $\lambda$ for the normalization constraint:

$$
\mathcal{L}(d, \lambda) = \sum_a d(a|s)\left[r(s,a) + \gamma \mathbb{E}_{s'}[v(s')]\right] - \alpha \sum_a d(a|s)\log d(a|s) - \lambda\left(\sum_a d(a|s) - 1\right)
$$

Taking the partial derivative with respect to $d(a|s)$ and setting it to zero yields the first-order optimality condition:

$$
\frac{\partial \mathcal{L}}{\partial d(a|s)} = r(s,a) + \gamma \mathbb{E}_{s'}[v(s')] - \alpha(1 + \log d(a|s)) - \lambda = 0
$$

Solving for $d^*(a|s)$:

$$
d^*(a|s) = \exp\left(\frac{1}{\alpha}\left(r(s,a) + \gamma \mathbb{E}_{s'}[v(s')] - \lambda\right)\right)
$$

Using the normalization constraint $\sum_a d^*(a|s) = 1$ to eliminate $\lambda$, we obtain the softmax policy:

$$
d^*(a|s) = \frac{\exp\left(\frac{1}{\alpha}(r(s,a) + \gamma \mathbb{E}_{s'}[v(s')])\right)}{\sum_{a'} \exp\left(\frac{1}{\alpha}(r(s,a') + \gamma \mathbb{E}_{s'}[v(s')])\right)}
$$

Substituting this optimal decision rule back into the entropy-regularized objective yields the smooth Bellman equation. Setting $\beta = 1/\alpha$ (the inverse temperature), we obtain:

$$
\begin{align*}
v(s) &= \sum_a d^*(a|s)[r(s,a) + \gamma \mathbb{E}_{s'}[v(s')]] + \alpha \mathcal{H}(d^*(\cdot|s)) \\
&= \frac{1}{\beta} \log \sum_a \exp\left(\beta(r(s,a) + \gamma \mathbb{E}_{s'}[v(s')])\right)
\end{align*}
$$

This establishes the equivalence between entropy regularization and smooth Bellman equations discussed in the [smoothing chapter](smoothing.md).
### Smooth Bellman Operator for Q-Functions

The smooth Bellman optimality operator for Q-functions is defined as:

$$
(\mathrm{L}_\alpha q)(s,a) = r(s,a) + \gamma \mathbb{E}_{s'}\left[\frac{1}{\beta} \log \sum_{a'} \exp(\beta \cdot q(s',a'))\right]
$$

where $\beta = 1/\alpha$ is the inverse temperature. This operator maintains the contraction property of its standard counterpart, guaranteeing a unique fixed point $q^*$. 

The optimal policy derived from the optimal Q-function takes the softmax form:

$$
\pi^*(a|s) = \frac{\exp(\beta \cdot q^*(s,a))}{\sum_{a'} \exp(\beta \cdot q^*(s,a'))}
$$

The optimal value function can be recovered from the optimal Q-function using the log-sum-exp:

$$
v^*(s) = \frac{1}{\beta} \log \sum_a \exp(\beta \cdot q^*(s,a))
$$

As $\beta \to \infty$ (equivalently $\alpha \to 0$), these equations recover the standard Bellman optimality equations with $\max$ operations and deterministic greedy policies.

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


However, an important question remains: how can we solve this optimization problem when it involves the intractable partition function $Z(s)$? To see this, recall that for two distributions p and q, the KL divergence takes the form $D_{KL}(p\|q) = \mathbb{E}_{x \sim p}[\log p(x) - \log q(x)]$. Denote the target Boltzmann distribution based on our current Q-estimate as:

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

One last challenge remains: $\phi$ appears in the distribution underlying the inner expectation, as well as in the integrand. This setting departs from standard empirical risk minimization (ERM) in supervised learning where the distribution of the data (e.g., cats and dogs in image classification) remains fixed regardless of model parameters. Here, however, the "data" (our sampled actions) depends directly on the parameters $\phi$ we're trying to optimize.

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

The methods we have seen so far (NFQCA, DDPG, TD3, SAC) all amortize action selection by learning a policy network that directly outputs actions. This amortizes the $\arg\max$ operation across states. But there is another axis of amortization: across time.

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

