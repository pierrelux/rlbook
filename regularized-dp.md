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

# Entropy-Regularized Dynamic Programming

Infinite-horizon dynamic programming selects actions with a hard maximum. That
operator is nondifferentiable at ties and produces a deterministic greedy
choice. Can a modified control objective produce both a smooth Bellman operator
and a stochastic optimal policy? Convex regularization supplies the link.

While the standard Bellman optimality equations use the max operator to determine the best action, an alternative formulation known as the smooth or soft Bellman optimality equations replaces it with a smooth approximation of the maximum. The history of these equations runs across two communities that rarely read each other. They appear in the work of the structural econometrician John Rust on the replacement of bus engines {cite}`rust1987optimal`, together with an inverse problem and an algorithm to solve it. {cite:t}`NgRussell2000`, who are usually credited with introducing inverse reinforcement learning, cite Rust, but the connection was then largely lost. {cite:t}`ziebart2008maximum` rediscovered the same equations as maximum entropy inverse reinforcement learning, named after the maximum entropy principle. {cite:t}`haarnoja2017reinforcement` built on Ziebart to obtain soft Q-learning, and soft actor-critic {cite}`haarnoja2018soft`, one of the most widely used deep RL algorithms for continuous control, followed a year later.

In the infinite-horizon setting, the smooth Bellman optimality equations take the form:

$$ v_\gamma^\star(s) = \frac{1}{\beta} \log \sum_{a \in A_s} \exp\left(\beta\left(r(s, a) + \gamma \sum_{j \in S} p(j | s, a) v_\gamma^\star(j)\right)\right) $$

Adopting an operator-theoretic perspective, we can define a nonlinear operator $\mathrm{L}_\beta$ such that the smooth value function of an MDP is then the solution to the following fixed-point equation:

$$ (\mathrm{L}_\beta \mathbf{v})(s) = \frac{1}{\beta} \log \sum_{a \in \mathcal{A}_s} \exp\left(\beta\left(r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right)\right) $$

As $\beta \to \infty$, $\mathrm{L}_\beta$ converges to the standard Bellman operator $\Bellman$. Furthermore, it can be shown that the smooth Bellman operator is a contraction mapping in the supremum norm, and therefore has a unique fixed point. However, as opposed to the usual "hard" setting, the fixed point of $\mathrm{L}_\beta$ is associated with the value function of an optimal stochastic policy defined by the softmax distribution:

   $$ \pi(a|s) = \frac{\exp\left(\beta\left(r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v_\gamma^\star(j)\right)\right)}{\sum_{a' \in \mathcal{A}_s} \exp\left(\beta\left(r(s,a') + \gamma \sum_{j \in \mathcal{S}} p(j|s,a') v_\gamma^\star(j)\right)\right)} $$

Despite the confusing terminology, the above "softmax" policy is the smooth counterpart of the argmax in the original optimality equation, not of the max: it is a soft-argmax. The smooth counterpart of the max is the log-sum-exp. The same policy is also called a Boltzmann or Gibbs policy. As the temperature $1/\beta$ goes to zero, the softmax concentrates on the maximizing actions and we recover the greedy policy and the hard Bellman equation.

### Stochastic Policies: Exploration and the Partition Function

Moving from the argmax to the softmax turns deterministic policies into stochastic ones, which is both useful and costly. It is useful because noise in the actions is a way to explore. In RL the agent usually gathers its own data, and if it always took the action it currently believes best it would never learn about the others. The problem is not specific to RL. An engineer identifying a model of a plant must also choose inputs that reveal the dynamics, and in system identification this is called *exciting* the system: a persistently exciting input is one rich enough for the parameters to be identifiable from the data.

It is costly when the action set is continuous. The softmax policy is then a density $\pi(a \mid s) \propto \exp(\beta q(s,a))$ whose normalizing constant, the partition function $\int \exp(\beta q(s,a))\, da$, is an integral with no closed form in general. The log-sum-exp value has the same problem, since it is the logarithm of that integral. This is the intractable partition function familiar from energy-based models and probabilistic graphical models. Numerical quadrature is an option in low dimension but is rarely used. {cite:t}`haarnoja2017reinforcement` sampled from the softmax policy over continuous actions with amortized Stein variational gradient descent, which proved expensive; soft actor-critic {cite}`haarnoja2018soft` instead fits a tractable parametric policy to the softmax, an idea we return to in the chapters on fitted value methods and policy learning.



This formulation is interesting for several reasons. First, smoothness is a desirable property from an optimization standpoint. Unlike $\gamma$, we view $\beta$ as a hyperparameter of our algorithm, which we can control to achieve the desired level of accuracy.

Second, we presented the equations pragmatically, by replacing the max with the log-sum-exp (a smooth maximum) and the argmax with the softmax (a smooth argmax). The same equations come out of two other starting points, and this chapter follows all three routes:

1. **Smoothing the operator.** Replace every max by a log-sum-exp, as above. This is the route of the next subsection, and the justification is computational: a smooth operator has gradients everywhere.
2. **Noise on the rewards.** Keep the hard Bellman equation, but suppose that each action's reward is perturbed by independent Gumbel noise that the decision maker sees and the modeler does not. Integrating the noise out gives the log-sum-exp ({ref}`sec-gumbel-rewards`). This is how {cite:t}`rust1987optimal` motivated the equations, following the discrete choice models of econometrics.
3. **Entropy regularization.** Keep the rewards, but add a bonus proportional to the entropy of the policy in every state. The regularized Bellman equation is again the log-sum-exp equation ({ref}`sec-regularized-mdps`). This is the reading behind maximum entropy RL and soft actor-critic.

Each route brings its own tools: smoothness gives differentiability, the noise model gives a likelihood for observed behavior, and the regularization view gives convex duality. A fourth reading, close to the third, starts from the energy-based models of supervised learning, where a softmax turns unnormalized scores into a distribution; this is essentially what {cite:t}`ziebart2008maximum` did. It connects to probabilistic graphical models, where finding an optimal smooth policy becomes an inference problem. This is the idea of control as inference, whose converse, inference as control, is now used to derive samplers and amortized inference methods with RL {cite}`levine2018reinforcement`.

### Smooth Value Iteration Algorithm

Before iterating $\mathrm{L}_\beta$, we should check that iterating it is the right thing to do, that is, that it is a contraction. It is, in the sup-norm and with the same factor $\gamma$ as the hard Bellman operator. The proof uses the same three facts as for $\Bellman$. The log-sum-exp is monotone, it commutes with adding a constant to all of its arguments, and $\mathrm{L}_\beta$ maps a uniform shift $c\mathbf{1}$ of $v$ into a shift $\gamma c\mathbf{1}$ of $\mathrm{L}_\beta v$. From $v \le u + \|v-u\|_\infty\mathbf{1}$ we get $\mathrm{L}_\beta v \le \mathrm{L}_\beta u + \gamma\|v-u\|_\infty\mathbf{1}$, and exchanging $u$ and $v$ gives the contraction. A word of caution about the converse: failing to show that an operator is a contraction does not mean that its fixed point fails to exist or that no algorithm finds it. Contraction is a convenient sufficient condition, and other fixed-point theorems cover other cases.

Successive approximation applied to $\mathrm{L}_\beta$ is **smooth value iteration**. It has the structure of value iteration, with a log-sum-exp wherever value iteration had a max and a softmax policy at the end where value iteration returned a greedy one.

```{prf:algorithm} Smooth Value Iteration
:label: smooth-value-iteration

**Input:** MDP $(S, A, r, p, \gamma)$, inverse temperature $\beta > 0$, tolerance $\varepsilon > 0$

**Output:** Value function $v$ within $\varepsilon/2$ of $v^\star_\beta$ and stochastic policy $\pi$

1. Initialize $v(s) \leftarrow 0$ for all $s \in S$
2. **repeat**
3. $\quad$ **for** each state $s \in S$ and action $a \in A_s$ **do**
4. $\quad\quad q(s,a) \leftarrow r(s,a) + \gamma \sum_{j \in S} p(j|s,a) v(j)$
5. $\quad$ **end for**
6. $\quad v_{\text{new}}(s) \leftarrow \frac{1}{\beta} \log \sum_{a \in A_s} \exp(\beta \cdot q(s,a))$ for all $s \in S$
7. $\quad \Delta \leftarrow \max_s |v_{\text{new}}(s) - v(s)|$
8. $\quad v \leftarrow v_{\text{new}}$
9. **until** $\Delta < \frac{\varepsilon(1-\gamma)}{2\gamma}$
10. Extract policy: compute $q(s,a)$ from $v$ as in line 4, and set $\pi(a|s) \leftarrow \frac{\exp(\beta \cdot q(s,a))}{\sum_{a' \in A_s} \exp(\beta \cdot q(s,a'))}$
11. **return** $v, \pi$
```

**Differences from standard value iteration:**
- Line 6 uses $\frac{1}{\beta} \log \sum_a \exp(\beta \cdot q(s,a))$ instead of $\max_a q(s,a)$
- Line 10 extracts a stochastic policy using softmax instead of a deterministic argmax policy
- As $\beta \to \infty$, the algorithm converges to standard value iteration
- Lower $\beta$ values produce more stochastic policies with higher entropy

**What the stopping rule guarantees.** Because $\mathrm{L}_\beta$ is a $\gamma$-contraction, the a posteriori bound of the Banach fixed-point theorem applies unchanged, and the stopping test of line 9 guarantees $\|v - v^\star_\beta\|_\infty \le \varepsilon/2$. The proof that value iteration returns an $\varepsilon$-optimal policy also carries over, with one change of meaning. The softmax policy satisfies $\mathrm{L}_{\pi,\Omega} v = \mathrm{L}_\beta v$ for the entropy-regularized evaluation operator of {ref}`sec-regularized-operators`, with regularizer $\Omega(\pi(\cdot|s)) = \frac{1}{\beta}\sum_a \pi(a|s)\log\pi(a|s)$,, just as a greedy policy satisfies $\mathrm{L}_\pi v = \Bellman v$. Repeating the proof of {prf:ref}`value-iteration-convergence` with these operators shows that the returned policy is $\varepsilon$-optimal *for the entropy-regularized objective*. How far is that from optimal for the original rewards? Since $\max_a x_a \le \frac{1}{\beta}\log\sum_a e^{\beta x_a} \le \max_a x_a + \frac{\log|\mathcal{A}|}{\beta}$, where $|\mathcal{A}|$ is the largest number of actions in any state, the two fixed points satisfy

$$
v^*_\gamma \;\le\; v^\star_\beta \;\le\; v^*_\gamma + \frac{\log |\mathcal{A}|}{\beta(1-\gamma)}\mathbf{1},
$$

and the softmax policy returned by the algorithm loses at most $\varepsilon + \frac{\log|\mathcal{A}|}{\beta(1-\gamma)}$ against the optimal value of the unregularized MDP. The temperature therefore adds a bias that does not shrink with more iterations; it is the price of smoothness and of exploration.

(sec-gumbel-rewards)=
## Gumbel Noise on the Rewards

Why does adding independent Gumbel perturbations to action rewards produce a
log-sum-exp value before the perturbation is observed?

We can obtain the smooth Bellman equation by considering a setting in which we have Gumbel noise added to the reward function. This derivation provides theoretical insight and also connects to practical modeling scenarios where rewards have random perturbations.

### Step 1: Define the Augmented MDP with Gumbel Noise

At each time period and state $s$, we draw an **action-indexed shock vector**:

$$\boldsymbol{\epsilon}_t(s) = \big(\epsilon_t(s,a)\big)_{a \in \mathcal{A}_s}, \quad \text{where } \epsilon_t(s,a) \text{ i.i.d.} \sim \mathrm{Gumbel}(\mu_\epsilon, 1/\beta)$$

These shocks are independent across time periods, states, and actions, and are independent of the MDP transition dynamics $p(\cdot | s, a)$.

The **Gumbel distribution** with location parameter $\mu$ and scale parameter $1/\beta$ has probability density function:

$$ f(x; \mu, \beta) = \beta\exp\left(-\beta(x-\mu)-\exp(-\beta(x-\mu))\right) $$

To generate a Gumbel-distributed random variable, we can use inverse transform sampling: $X = \mu - \frac{1}{\beta} \ln(-\ln(U))$ where $U$ is uniform on $(0,1)$.

```{admonition} Zero-Mean Shocks
:class: tip
To ensure the shocks have zero mean, we set $\mu_\epsilon = -\gamma_E/\beta$ where $\gamma_E \approx 0.5772$ is the Euler-Mascheroni constant. This choice eliminates an additive constant that would otherwise appear in the smooth Bellman equation. For simplicity, we will adopt this convention throughout.
```

We now define an **augmented MDP** with:
- **Augmented state**: $\tilde{s} = (s, \boldsymbol{\epsilon})$ where $s \in \mathcal{S}$ and $\boldsymbol{\epsilon} \in \mathbb{R}^{|\mathcal{A}_s|}$
- **Augmented reward**: $\tilde{r}(\tilde{s}, a) = r(s,a) + \epsilon(a)$
- **Augmented transition**: $\tilde{p}(\tilde{s}' | \tilde{s}, a) = p(s' | s, a) \cdot p(\boldsymbol{\epsilon}')$

The transition factorizes because the next shock vector $\boldsymbol{\epsilon}'$ is drawn independently of the current state and action (conditional independence).

```{admonition} The Augmented State Space is Infinite-Dimensional
:class: warning
Even if the original state space $\mathcal{S}$ and action space $\mathcal{A}$ are finite, the augmented state space $\tilde{\mathcal{S}} = \mathcal{S} \times \mathbb{R}^{|\mathcal{A}|}$ is **uncountably infinite** because each shock vector $\boldsymbol{\epsilon}$ is a continuous random variable. Therefore:
- We cannot enumerate the augmented states
- Tabular dynamic programming methods do not apply directly
- The augmented value function $\tilde{v}(s, \boldsymbol{\epsilon})$ maps a continuous space to $\mathbb{R}$

**This motivates why we immediately marginalize over the shocks** to obtain a finite-dimensional representation.
```

### Step 2: The Hard Bellman Equation on the Augmented State Space

The Bellman optimality equation for the augmented MDP is:

$$ \tilde{v}(s, \boldsymbol{\epsilon}) = \max_{a \in \mathcal{A}_s} \left\{ r(s,a) + \epsilon(a) + \gamma \mathbb{E}_{s', \boldsymbol{\epsilon}'}\left[\tilde{v}(s', \boldsymbol{\epsilon}') \mid s, a\right] \right\} $$

Here the expectation is over the **next augmented state** $(s', \boldsymbol{\epsilon}')$, which includes both the next state $s' \sim p(\cdot | s, a)$ and the next shock vector $\boldsymbol{\epsilon}' \sim p(\cdot)$.

This is a perfectly well-defined Bellman equation, and an optimal stationary policy exists:

$$\pi(s, \boldsymbol{\epsilon}) \in \operatorname{argmax}_{a \in \mathcal{A}_s} \left\{ r(s,a) + \epsilon(a) + \gamma \mathbb{E}_{s', \boldsymbol{\epsilon}'}\left[\tilde{v}(s', \boldsymbol{\epsilon}') \mid s, a\right] \right\}$$

However, this equation is **computationally intractable** because:
- The state space is continuous and infinite-dimensional
- The shocks are fresh each period
- We would need to solve for $\tilde{v}$ over an uncountable domain

**We never solve this equation directly.** Instead, we use it as a mathematical device to derive the smooth Bellman equation.

### Step 3: Define the Ex-Ante (Inclusive) Value Function

The idea here is to consider the **expected value before observing the current shocks**. We define what some authors in econometrics call the **inclusive value** or **ex-ante value**:

$$ v(s) := \mathbb{E}_{\boldsymbol{\epsilon}}\big[\tilde{v}(s, \boldsymbol{\epsilon})\big] $$

This is the value of being in state $s$ **before** we observe the current-period shock vector $\boldsymbol{\epsilon}$. 

```{admonition} Two Different Value Functions
:class: note
It is crucial to distinguish:
- $\tilde{v}(s, \boldsymbol{\epsilon})$: the value **after** observing shocks (conditional on $\boldsymbol{\epsilon}$), defined on the augmented state space
- $v(s)$: the value **before** observing shocks (marginalizing over $\boldsymbol{\epsilon}$), defined on the original state space

The function $v(s)$ is what we actually compute and care about. The augmented value $\tilde{v}$ exists only as a proof device.
```

### Step 4: Separate the Deterministic and Random Components

Now we take the expectation of the augmented Bellman equation with respect to the **current shocks only** (everything that does not depend on the current $\boldsymbol{\epsilon}$ can be pulled out).

First, note that by the law of iterated expectations and independence of shocks across time:

$$ \mathbb{E}_{\boldsymbol{\epsilon}'}\big[\tilde{v}(s', \boldsymbol{\epsilon}')\big] = v(s') $$

This follows from our definition of $v$ and the fact that the next shock is independent of everything else.

Now define the **deterministic part** of the right-hand side:

$$ x_a(s) := r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j) $$

This is the expected return from taking action $a$ in state $s$ **without the shock**. Using this notation, the augmented Bellman equation becomes:

$$ \tilde{v}(s, \boldsymbol{\epsilon}) = \max_{a \in \mathcal{A}_s} \left\{ x_a(s) + \epsilon(a) \right\} $$

Taking the expectation over $\boldsymbol{\epsilon}$ on both sides:

$$ v(s) = \mathbb{E}_{\boldsymbol{\epsilon}}\left[\max_{a \in \mathcal{A}_s} \left\{ x_a(s) + \epsilon(a) \right\}\right] $$

```{admonition} Expectation of a Max, Not Max of an Expectation
:class: important
Notice carefully: we have $\mathbb{E}[\max(\cdot)]$, **not** $\max \mathbb{E}[\cdot]$. We are **not** swapping max and expectation. 

The expression $\mathbb{E}_{\boldsymbol{\epsilon}}[\max_a \{x_a + \epsilon(a)\}]$ is the expected value of the maximum of Gumbel-perturbed utilities. The Gumbel random utility identity evaluates this quantity in closed form.
```

### Step 5: Apply the Gumbel Random Utility Identity

We now invoke a result from extreme value theory:

```{prf:lemma} Gumbel Random Utility Identity
:label: gumbel-random-utility

Let $\epsilon_1, \ldots, \epsilon_m$ be i.i.d. $\mathrm{Gumbel}(\mu_\epsilon, 1/\beta)$ random variables. For any deterministic values $x_1, \ldots, x_m \in \mathbb{R}$:

$$ \max_{i=1,\ldots,m} \{x_i + \epsilon_i\} \overset{d}{=} \frac{1}{\beta} \log \sum_{i=1}^m \exp(\beta x_i) + \zeta $$

where $\zeta \sim \mathrm{Gumbel}(\mu_\epsilon, 1/\beta)$ (same distribution as the original shocks).

Taking expectations:

$$ \mathbb{E}\left[\max_{i=1,\ldots,m} \{x_i + \epsilon_i\}\right] = \frac{1}{\beta} \log \sum_{i=1}^m \exp(\beta x_i) + \mu_\epsilon + \frac{\gamma_E}{\beta} $$

where $\gamma_E \approx 0.5772$ is the Euler-Mascheroni constant.

**With mean-zero shocks** ($\mu_\epsilon = -\gamma_E/\beta$), the constant term vanishes:

$$ \mathbb{E}\left[\max_{i=1,\ldots,m} \{x_i + \epsilon_i\}\right] = \frac{1}{\beta} \log \sum_{i=1}^m \exp(\beta x_i) $$
```

Applying this identity to our problem (with mean-zero shocks):

$$ v(s) = \mathbb{E}_{\boldsymbol{\epsilon}}\left[\max_{a \in \mathcal{A}_s} \{x_a(s) + \epsilon(a)\}\right] = \frac{1}{\beta} \log \sum_{a \in \mathcal{A}_s} \exp(\beta x_a(s)) $$

Substituting the definition of $x_a(s)$:

$$ v(s) = \frac{1}{\beta} \log \sum_{a \in \mathcal{A}_s} \exp\left(\beta\left(r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right)\right) $$

We have arrived at the **smooth Bellman equation**.

### Step 6: Summary of the Derivation

To recap the logical flow:

1. We constructed an augmented MDP with state $(s, \boldsymbol{\epsilon})$ where shocks perturb rewards
2. We wrote the standard Bellman equation for this augmented MDP (hard max, but over an infinite-dimensional state space)
3. We defined the ex-ante value $v(s) = \mathbb{E}_{\boldsymbol{\epsilon}}[\tilde{v}(s, \boldsymbol{\epsilon})]$ to eliminate the continuous shock component
4. We separated deterministic and random terms: $\tilde{v}(s, \boldsymbol{\epsilon}) = \max_a \{x_a(s) + \epsilon(a)\}$
5. We applied the Gumbel identity to evaluate $\mathbb{E}_{\boldsymbol{\epsilon}}[\max_a \{\cdots\}]$ in closed form as a log-sum-exp

The augmented MDP with shocks exists **only as a mathematical device**. We never approximate $\tilde{v}$, never discretize $\boldsymbol{\epsilon}$, and never enumerate the augmented state space. The only computational object we work with is $v(s)$ on the original (finite) state space, which satisfies the smooth Bellman equation.

### Deriving the Optimal Smooth Policy

Now that we have derived the smooth value function, we can also obtain the corresponding optimal policy. The question is: **what policy should we follow in the original MDP (without explicitly conditioning on shocks)?**

In the augmented MDP, the optimal policy is deterministic but depends on the shock realization:

$$\pi(s, \boldsymbol{\epsilon}) \in \operatorname{argmax}_{a \in \mathcal{A}_s} \left\{ r(s,a) + \epsilon(a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j) \right\}$$

However, we want a policy for the **original** state space $s$ (not the augmented state). We obtain this by **marginalizing over the current shocks**, essentially asking: "what is the probability that action $a$ is optimal when we average over all possible shock realizations?"

Define an indicator function:

$$ I_a(\boldsymbol{\epsilon}) = \begin{cases} 
   1 & \text{if } a \in \operatorname{argmax}_{a' \in \mathcal{A}_s} \left\{ x_{a'}(s) + \epsilon(a') \right\} \\
   0 & \text{otherwise}
   \end{cases} $$

where $x_a(s) = r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)$ as before.

The ex-ante probability that action $a$ is optimal at state $s$ is:

$$ \pi(a|s) = \mathbb{E}_{\boldsymbol{\epsilon}}[I_a(\boldsymbol{\epsilon})] = \mathbb{P}_{\boldsymbol{\epsilon}}\left(a \in \operatorname{argmax}_{a'} \left\{ x_{a'}(s) + \epsilon(a') \right\}\right) $$

This is the probability that action $a$ achieves the maximum when utilities are perturbed by Gumbel noise.

```{prf:lemma} Gumbel-Max Probability (Softmax)
:label: gumbel-softmax

Let $\epsilon_1, \ldots, \epsilon_m$ be i.i.d. $\mathrm{Gumbel}(\mu_\epsilon, 1/\beta)$ random variables. For any deterministic values $x_1, \ldots, x_m \in \mathbb{R}$, the probability that index $i$ achieves the maximum is:

$$ \mathbb{P}\left(i \in \operatorname{argmax}_j \{x_j + \epsilon_j\}\right) = \frac{\exp(\beta x_i)}{\sum_{j=1}^m \exp(\beta x_j)} $$

This holds regardless of the location parameter $\mu_\epsilon$.
```

Applying this result to our problem:

$$ \pi(a|s) = \frac{\exp\left(\beta x_a(s)\right)}{\sum_{a' \in \mathcal{A}_s} \exp\left(\beta x_{a'}(s)\right)} = \frac{\exp\left(\beta\left(r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)v(j)\right)\right)}{\sum_{a' \in \mathcal{A}_s} \exp\left(\beta\left(r(s,a') + \gamma \sum_{j \in \mathcal{S}} p(j|s,a')v(j)\right)\right)} $$

This is the **softmax policy** or **Gibbs/Boltzmann policy** with inverse temperature $\beta$.

**Properties:**
- As $\beta \to \infty$: the policy becomes deterministic, concentrating on the action(s) with highest $x_a(s)$ (recovers standard greedy policy)
- As $\beta \to 0$: the policy becomes uniform over all actions (maximum entropy)
- For finite $\beta > 0$: the policy is stochastic, with probability mass proportional to exponentiated Q-values

This completes the derivation: the smooth Bellman equation yields a value function $v(s)$, and the corresponding optimal policy is the softmax over Q-values.

<!-- ## Control as Inference Perspective

The smooth Bellman optimality equations can also be derived from probabilistic inference perspective. To see this, let's go back to the idea from the previous section in which we introduced an indicator function $I_a(\epsilon)$ to represent whether an action $a$ is optimal given a particular realization of the noise $\epsilon$:

$$ I_a(\epsilon) = \begin{cases} 
   1 & \text{if } a \in \operatorname{argmax}_{a' \in \mathcal{A}_s} \left\{ r(s,a') + \epsilon(a') + \gamma \mathbb{E}_{s', \epsilon'}\left[v_\gamma^\star(s',\epsilon')\mid s, a'\right] \right\} \\
   0 & \text{otherwise}
   \end{cases} $$

When we took the expectation over the noise $\epsilon$, we obtained a soft version of this indicator:

$$ \begin{align*}
\mathbb{E}_\epsilon[I_a(\epsilon)] &= \mathbb{P}\left(a \in \operatorname{argmax}_{a' \in \mathcal{A}_s} \left\{ r(s,a') + \epsilon(a') + \gamma \mathbb{E}_{s', \epsilon'}\left[v_\gamma^\star(s',\epsilon')\mid s, \epsilon, a'\right] \right\}\right) \\
&= \frac{\exp\left(\beta\left(r(s,a) + \gamma \mathbb{E}_{s'}\left[v_\gamma^\star(s')\mid s, a\right]\right)\right)}{\sum_{a' \in \mathcal{A}_s} \exp\left(\beta\left(r(s,a') + \gamma \mathbb{E}_{s'}\left[v_\gamma^\star(s')\mid s, a'\right]\right)\right)}
\end{align*} $$

Given this indicator function, we can "infer" the optimal action in any state. This is the intuition and starting point behind the control as inference perspective in which we directly define a continuous-valued "optimality" variable $O_t$ at each time step $t$. We define the probability of optimality given a state-action pair as:

$$ p(O_t = 1 | s_t, a_t) = \exp(\beta r(s_t, a_t)) $$

Building on this notion of soft optimality, we can formulate the MDP as a probabilistic graphical model. We define the following probabilities:

1. State transition probability: $p(s_{t+1} | s_t, a_t)$ (given by the MDP dynamics)
2. Prior policy: $p(a_t | s_t)$ (which we'll assume to be uniform for simplicity)
3. Optimality probability: $p(O_t = 1 | s_t, a_t) = \exp(\beta r(s_t, a_t))$

This formulation encodes the idea that more rewarding state-action pairs are more likely to be "optimal," which directly parallels the soft assignment of optimality we obtained by taking the expectation over the Gumbel noise.

The control problem can now be framed as an inference problem: we want to find the posterior distribution over actions given that all time steps are optimal:

$$ p(a_t | s_t, O_{1:T} = 1) $$

where $O_{1:T} = 1$ means $O_t = 1$ for all $t$ from 1 to T. 

### Message Passing 

To solve this inference problem, we can use a technique from probabilistic graphical models called message passing, specifically the belief propagation algorithm. Message passing is a way to efficiently compute marginal distributions in a graphical model by passing local messages between nodes. Messages are passed between nodes in both forward and backward directions. Each message represents a belief about the distribution of a variable, based on the information available to the sending node. After messages have been passed, each node updates its belief about its associated variable by combining all incoming messages.

In our specific case, we're particularly interested in the backward messages, which propagate information about future optimality backwards in time. Let's define the backward message $\beta_t(s_t)$ as:

$$ \beta_t(s_t) = p(O_{t:T} = 1 | s_t) $$

This represents the probability of optimality for all future time steps given the current state. We can compute this recursively:

$$ \beta_t(s_t) = \sum_{a_t} p(a_t | s_t) p(O_t = 1 | s_t, a_t) \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) \beta_{t+1}(s_{t+1}) $$


Taking the log and assuming a uniform prior over actions, we get:

$$ \log \beta_t(s_t) = \log \sum_{a_t} \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) \exp(\beta (r(s_t, a_t) + \gamma v(_{t+1}) + \frac{1}{\beta} \log \beta_{t+1}(s_{t+1}))) $$

If we define the soft value function as $V_t(s_t) = \frac{1}{\beta} \log \beta_t(s_t)$, we can rewrite the above equation as:

$$ V_t(s_t) = \frac{1}{\beta} \log \sum_{a_t} \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) \exp(\beta (r(s_t, a_t) + \gamma V_{t+1}(s_{t+1}))) $$

This is exactly the smooth Bellman equation we derived earlier, but now interpreted as the result of probabilistic inference in a graphical model.

### Deriving the Optimal Policy

The backward message recursion we derived earlier assumes a uniform prior policy $p(a_t | s_t)$. However, our goal is to find an optimal policy. We can extract this optimal policy efficiently by computing the posterior distribution over actions given our backward messages.

Starting from the definition of conditional probability and applying Bayes' rule, we can write:

$$ \begin{align}
p(a_t | s_t, O_{1:T} = 1) &= \frac{p(O_{1:T} = 1 | s_t, a_t) p(a_t | s_t)}{p(O_{1:T} = 1 | s_t)} \\
&\propto p(a_t | s_t) p(O_t = 1 | s_t, a_t) p(O_{t+1:T} = 1 | s_t, a_t) \\
&= p(a_t | s_t) p(O_t = 1 | s_t, a_t) \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) \beta_{t+1}(s_{t+1})
\end{align} $$

Here, $\beta_{t+1}(s_{t+1}) = p(O_{t+1:T} = 1 | s_{t+1})$ is our backward message.

Now, let's substitute our definitions for the optimality probability and the soft value function:

$$ \begin{align}
p(a_t | s_t, O_{1:T} = 1) &\propto p(a_t | s_t) \exp(\beta r(s_t, a_t)) \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) \exp(\beta \gamma V_{t+1}(s_{t+1})) \\
&= p(a_t | s_t) \exp(\beta (r(s_t, a_t) + \gamma \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) V_{t+1}(s_{t+1})))
\end{align} $$

After normalization, and assuming a uniform prior $p(a_t | s_t)$, we obtain the randomized decision rule:

$$ d(a_t | s_t) = \frac{\exp(\beta (r(s_t, a_t) + \gamma \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) V_{t+1}(s_{t+1})))}{\sum_{a'_t} \exp(\beta (r(s_t, a'_t) + \gamma \sum_{s_{t+1}} p(s_{t+1} | s_t, a'_t) V_{t+1}(s_{t+1})))} $$ -->

(sec-regularized-mdps)=
## Regularized Markov Decision Processes

Which convex penalty on action distributions generates a desired smooth
maximum through conjugate duality?

Regularized MDPs {cite}`geist2019` provide another perspective on how the smooth Bellman equations come to be. This framework offers a more general approach in which we seek to find optimal policies under the infinite horizon criterion while also accounting for a regularizer that influences the kind of policies we try to obtain.

Let's set up some necessary notation. First, recall that the policy evaluation operator for a stationary policy with decision rule $\pi$ is defined as:

$$ \BellmanPi \mathbf{v} = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v} $$

where $\mathbf{r}_\pi$ is the expected reward vector under policy $\pi$, $\gamma$ is the discount factor, and $\mathbf{P}_\pi$ is the state transition probability matrix under $\pi$. A complementary object to the value function is the q-function (or Q-factor) representation:

$$ \begin{align*}
q_\gamma^{\pi}(s, a) &= r(s, a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v_\gamma^{\pi}(j) \\
v_\gamma^{\pi}(s) &= \sum_{a \in \mathcal{A}_s} \pi(a | s) q_\gamma^{\pi}(s, a) 
\end{align*} $$

The policy evaluation operator can then be written in terms of the q-function as:

$$ [\BellmanPi v](s) = \langle \pi(\cdot | s), q(s, \cdot) \rangle $$

### Legendre-Fenchel Transform

The workhorse behind the theory of regularized MDPs is the Legendre-Fenchel transform, also known as the convex conjugate. For a strongly convex function $\Omega: \Delta_{\mathcal{A}} \rightarrow \mathbb{R}$, its Legendre-Fenchel transform $\Omega^*: \mathbb{R}^{\mathcal{A}} \rightarrow \mathbb{R}$ is defined as:

$$ \Omega^*(q(s, \cdot)) = \max_{\pi(\cdot|s) \in \Delta_{\mathcal{A}}} \langle \pi(\cdot | s), q(s, \cdot) \rangle - \Omega(\pi(\cdot | s)) $$

An important property of this transform is that it has a unique maximizing argument, given by the gradient of $\Omega^*$. This gradient is Lipschitz and satisfies:

$$ \nabla \Omega^*(q(s, \cdot)) = \arg\max_\pi \langle \pi(\cdot | s), q(s, \cdot) \rangle - \Omega(\pi(\cdot | s)) $$

An important example of a regularizer is the negative entropy, which gives rise to the smooth Bellman equations as we are about to see. 

(sec-regularized-operators)=
## Regularized Bellman Operators

How do the regularizer and its convex conjugate alter Bellman evaluation,
policy improvement, and their contraction properties?

With these concepts in place, we can now define the regularized Bellman operators:

1. **Regularized Policy Evaluation Operator** $(\mathrm{L}_{\pi,\Omega})$:

   $$ [\mathrm{L}_{\pi,\Omega} v](s) = \langle q(s,\cdot), \pi(\cdot | s) \rangle - \Omega(\pi(\cdot | s)) $$

2. **Regularized Bellman Optimality Operator** $(\mathrm{L}_\Omega)$:
           
   $$ [\mathrm{L}_\Omega v](s) = [\max_\pi \mathrm{L}_{\pi,\Omega} v ](s) = \Omega^*(q(s, \cdot)) $$

It can be shown that the addition of a regularizer in these regularized operators still preserves the contraction properties, and therefore the existence of a solution to the optimality equations and the convergence of successive approximation.

The regularized value function of a stationary policy with decision rule $\pi$, denoted by $v_{\pi,\Omega}$, is the unique fixed point of the operator equation:

$$\text{find $v$ such that } \enspace v = \mathrm{L}_{\pi,\Omega} v$$

Under the usual assumptions on the discount factor and the boundedness of the reward, the value of a policy can also be found in closed form by solving for $\mathbf{v}$ in the linear system of equations:

$$ (\mathbf{I} - \gamma \mathbf{P}_\pi) \mathbf{v} =  \mathbf{r}_\pi - \boldsymbol{\Omega}_\pi $$

where $[\boldsymbol{\Omega}_\pi](s) = \Omega(\pi(\cdot|s))$ is the vector of regularization terms at each state.

The associated state-action value function $q_{\pi,\Omega}$ is given by:

$$\begin{align*}
q_{\pi,\Omega}(s, a) &= r(s, a) + \sum_{j \in \mathcal{S}} \gamma p(j|s,a) v_{\pi,\Omega}(j) \\
v_{\pi,\Omega}(s) &= \sum_{a \in \mathcal{A}_s} \pi(a | s) q_{\pi,\Omega}(s, a) - \Omega(\pi(\cdot | s))
\end{align*} $$

The regularized optimal value function $v^*_\Omega$ is then the unique fixed point of $\mathrm{L}_\Omega$ in the fixed point equation:

$$\text{find $v$ such that } v = \mathrm{L}_\Omega v$$

The associated state-action value function $q^*_\Omega$ is given by:

$$ \begin{align*}
q^*_\Omega(s, a) &= r(s, a) + \sum_{j \in \mathcal{S}} \gamma p(j|s,a) v^*_\Omega(j) \\
v^*_\Omega(s) &= \Omega^*(q^*_\Omega(s, \cdot))\end{align*} $$

An important result in the theory of regularized MDPs is that there exists a unique optimal regularized policy. Specifically, if $\pi^*_\Omega$ is a conserving decision rule (i.e., $\pi^*_\Omega = \arg\max_\pi \mathrm{L}_{\pi,\Omega} v^*_\Omega$), then the randomized stationary policy $\boldsymbol{\pi} = \mathrm{const}(\pi^*_\Omega)$ is the unique optimal regularized policy.

In practice, once we have found $v^*_\Omega$, we can derive the optimal decision rule by taking the gradient of the convex conjugate evaluated at the optimal action-value function:

$$ \pi^*(\cdot | s) = \nabla \Omega^*(q^*_\Omega(s, \cdot)) $$

### Recovering the Smooth Bellman Equations

Under this framework, we can recover the smooth Bellman equations by choosing $\Omega$ to be the negative entropy, and obtain the softmax policy as the gradient of the convex conjugate. Let's show this explicitly:

1. Using the negative entropy regularizer:

   $$ \Omega(d(\cdot|s)) = \sum_{a \in \mathcal{A}_s} d(a|s) \ln d(a|s) $$

2. The convex conjugate:

   $$ \Omega^*(q(s, \cdot)) = \ln \sum_{a \in \mathcal{A}_s} \exp q(s,a) $$

3. Now, let's write out the regularized Bellman optimality equation:

   $$ v^*_\Omega(s) = \Omega^*(q^*_\Omega(s, \cdot)) $$

4. Substituting the expressions for $\Omega^*$ and $q^*_\Omega$:

   $$ v^*_\Omega(s) = \ln \sum_{a \in \mathcal{A}_s} \exp \left(r(s, a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v^*_\Omega(j)\right) $$

This matches the form of the smooth Bellman equation we derived earlier, with the log-sum-exp operation replacing the max operation of the standard Bellman equation.

Furthermore, the optimal policy is given by the gradient of $\Omega^*$:

$$ d^*(a|s) = \nabla \Omega^*(q^*_\Omega(s, \cdot)) = \frac{\exp(q^*_\Omega(s,a))}{\sum_{a' \in \mathcal{A}_s} \exp(q^*_\Omega(s,a'))} $$

This is the familiar softmax policy we encountered in the smooth MDP setting.

### Smooth Policy Iteration Algorithm

Now that we've seen how the regularized MDP framework leads to smooth Bellman equations, we present smooth policy iteration. Unlike value iteration which directly iterates the Bellman operator, policy iteration alternates between policy evaluation and policy improvement steps.

```{prf:algorithm} Smooth Policy Evaluation
:label: smooth-policy-evaluation

**Input:** MDP $(S, A, r, p, \gamma)$, policy $\pi$, inverse temperature $\beta > 0$, tolerance $\epsilon > 0$

**Output:** Value function $v^\pi$ for policy $\pi$

1. Initialize $v(s) \leftarrow 0$ for all $s \in S$
2. Set $\alpha \leftarrow 1/\beta$
3. **repeat**
4. $\quad \Delta \leftarrow 0$
5. $\quad$ **for** each state $s \in S$ **do**
6. $\quad\quad v_{\text{old}} \leftarrow v(s)$
7. $\quad\quad$ **for** each action $a \in A_s$ **do**
8. $\quad\quad\quad q(s,a) \leftarrow r(s,a) + \gamma \sum_{j \in S} p(j|s,a) v(j)$
9. $\quad\quad$ **end for**
10. $\quad\quad$ Compute expected Q-value: $\bar{q} \leftarrow \sum_{a \in A_s} \pi(a|s) \cdot q(s,a)$
11. $\quad\quad$ Compute policy entropy: $H \leftarrow -\sum_{a \in A_s} \pi(a|s) \log \pi(a|s)$
12. $\quad\quad v(s) \leftarrow \bar{q} + \alpha H$
13. $\quad\quad \Delta \leftarrow \max(\Delta, |v(s) - v_{\text{old}}|)$
14. $\quad$ **end for**
15. **until** $\Delta < \epsilon$
16. **return** $v$
```

```{prf:algorithm} Smooth Policy Iteration
:label: policy-iteration-smooth

**Input:** MDP $(S, A, r, p, \gamma)$, inverse temperature $\beta > 0$, tolerance $\epsilon > 0$

**Output:** Approximate optimal value function $v$ and stochastic policy $\pi$

1. Initialize $\pi(a|s) \leftarrow 1/|A_s|$ for all $s \in S, a \in A_s$ (uniform policy)
2. **repeat**
3. $\quad$ **Policy Evaluation:**
4. $\quad\quad$ $v \leftarrow$ SmoothPolicyEvaluation($S, A, r, p, \gamma, \pi, \beta, \epsilon$)
5. $\quad$ **Policy Improvement:**
6. $\quad$ policy_stable $\leftarrow$ true
7. $\quad$ **for** each state $s \in S$ **do**
8. $\quad\quad \pi_{\text{old}}(\cdot|s) \leftarrow \pi(\cdot|s)$
9. $\quad\quad$ **for** each action $a \in A_s$ **do**
10. $\quad\quad\quad q(s,a) \leftarrow r(s,a) + \gamma \sum_{j \in S} p(j|s,a) v(j)$
11. $\quad\quad$ **end for**
12. $\quad\quad$ **for** each action $a \in A_s$ **do**
13. $\quad\quad\quad \pi(a|s) \leftarrow \frac{\exp(\beta \cdot q(s,a))}{\sum_{a' \in A_s} \exp(\beta \cdot q(s,a'))}$
14. $\quad\quad$ **end for**
15. $\quad\quad$ **if** $\|\pi(\cdot|s) - \pi_{\text{old}}(\cdot|s)\| > \epsilon$ **then**
16. $\quad\quad\quad$ policy_stable $\leftarrow$ false
17. $\quad\quad$ **end if**
18. $\quad$ **end for**
19. **until** policy_stable
20. **return** $v, \pi$
```

**Key properties of smooth policy iteration:**

1. **Entropy-regularized evaluation**: The policy evaluation step (line 12 of {prf:ref}`smooth-policy-evaluation`) accounts for the entropy bonus $\alpha H(\pi(\cdot|s))$ where $\alpha = 1/\beta$
2. **Stochastic policy improvement**: The policy improvement step (lines 12-14 of {prf:ref}`policy-iteration-smooth`) uses softmax instead of deterministic argmax, producing a stochastic policy
3. **Temperature parameter**: 
   - Higher $\beta$ produces policies closer to deterministic (lower entropy)
   - Lower $\beta$ produces more stochastic policies (higher entropy)
   - As $\beta \to \infty$, the method recovers standard policy iteration
4. **Convergence**: Like standard policy iteration, this algorithm converges to the unique optimal regularized value function and policy
5. **Partial improvement**: As in the hard case, the improvement does not need to be applied in every state for the method to make progress. The policy-iteration loops hidden inside actor-critic methods make use of this.

(sec-smooth-pi-newton)=
### Smooth Policy Iteration Is Newton's Method

The link between policy iteration and the Newton-Kantorovich method ({ref}`sec-newton-kantorovich-pi`) is easier to see here than in the hard case, because $\mathrm{L}_\beta$ is differentiable everywhere and there are no ties to worry about. The derivative of the log-sum-exp with respect to its arguments is the softmax, so the Fréchet derivative of $\mathrm{L}_\beta$ at $v$ is

$$
\mathrm{L}_\beta'(v) = \gamma \mathbf{P}_{\pi_v}, \qquad \pi_v(a \mid s) = \frac{\exp(\beta q_v(s,a))}{\sum_{a'}\exp(\beta q_v(s,a'))}, \quad q_v(s,a) = r(s,a) + \gamma\sum_j p(j\mid s,a) v(j),
$$

the transition matrix of the softmax policy. A Newton step on $v - \mathrm{L}_\beta v = 0$ solves

$$
(\mathbf{I} - \gamma\mathbf{P}_{\pi_v})\, v_{\text{new}} = \mathrm{L}_\beta v - \gamma\mathbf{P}_{\pi_v} v.
$$

At the softmax policy the log-sum-exp equals the expected $q$ plus the entropy bonus, $\mathrm{L}_\beta v = \mathbf{r}_{\pi_v} + \gamma\mathbf{P}_{\pi_v} v + \frac{1}{\beta}\mathbf{H}_{\pi_v}$. The right-hand side is therefore $\mathbf{r}_{\pi_v} + \frac{1}{\beta}\mathbf{H}_{\pi_v}$, and the Newton step is exactly the entropy-regularized evaluation of the softmax policy. Smooth policy iteration is the Newton-Kantorovich method applied to the smooth Bellman equation, without the semismooth machinery that the hard max requires.

{cite:t}`rust1987optimal` solved his smooth Bellman equation this way. He did not call it policy iteration. He recognized an operator equation, a root-finding problem in a function space, and applied Newton-Kantorovich to it, after a few rounds of successive approximation to get close enough for Newton's method to converge quickly.

### Equivalence Between Smooth Bellman Equations and Entropy-Regularized MDPs

We have now seen two distinct ways to arrive at smooth Bellman equations. Earlier in this chapter, we introduced the logsumexp operator as a smooth approximation to the max operator, motivated by analytical tractability and the desire for differentiability. Just now, we derived the same equations through the lens of regularized MDPs, where we explicitly penalize the entropy of policies. These two perspectives are mathematically equivalent: solving the smooth Bellman equation with inverse temperature parameter $\beta$ yields exactly the same optimal value function and optimal policy as solving the entropy-regularized MDP with regularization strength $\alpha = 1/\beta$. The two formulations are not merely similar. They describe identical optimization problems.

To see this equivalence clearly, consider the standard MDP problem with rewards $r(s,a)$ and transition probabilities $p(j|s,a)$. The regularized MDP framework tells us to solve:

$$
\max_\pi \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \right] + \alpha \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t H(\pi(\cdot|s_t)) \right],
$$
where $H(\pi(\cdot|s)) = -\sum_a \pi(a|s) \ln \pi(a|s)$ is the entropy of the policy at state $s$, and $\alpha > 0$ is the entropy regularization strength.

We can rewrite this objective by absorbing the entropy term into a modified reward function. Define the entropy-augmented reward:

$$
\tilde{r}(s,a,\pi) = r(s,a) + \alpha H(\pi(\cdot|s)).
$$

However, this formulation makes the reward depend on the entire policy at each state, which is awkward. We can reformulate this more cleanly by expanding the entropy term. Recall that the entropy is:

$$
H(\pi(\cdot|s)) = -\sum_a \pi(a|s) \ln \pi(a|s).
$$

When we take the expectation over actions drawn from $\pi$, we have:

$$
\mathbb{E}_{a \sim \pi(\cdot|s)} [H(\pi(\cdot|s))] = \sum_a \pi(a|s) \left[-\sum_{a'} \pi(a'|s) \ln \pi(a'|s)\right] = -\sum_{a'} \pi(a'|s) \ln \pi(a'|s),
$$

since the entropy doesn't depend on which action is actually sampled. But we can also write this as:

$$
H(\pi(\cdot|s)) = -\sum_a \pi(a|s) \ln \pi(a|s) = \mathbb{E}_{a \sim \pi(\cdot|s)}[-\ln \pi(a|s)].
$$

This shows that adding $\alpha H(\pi(\cdot|s))$ to the expected reward at state $s$ is equivalent to adding $-\alpha \ln \pi(a|s)$ to the reward of taking action $a$ at state $s$. More formally:

$$
\begin{align*}
&\mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \right] + \alpha \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t H(\pi(\cdot|s_t)) \right] \\
&= \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \right] + \alpha \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t \mathbb{E}_{a_t \sim \pi(\cdot|s_t)}[-\ln \pi(a_t|s_t)] \right] \\
&= \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t \left( r(s_t, a_t) - \alpha \ln \pi(a_t|s_t) \right) \right].
\end{align*}
$$

The entropy bonus at each state, when averaged over the policy, becomes a per-action penalty proportional to the negative log probability of the action taken. This reformulation is more useful because the modified reward now depends only on the state, the action taken, and the probability assigned to that specific action by the policy, not on the entire distribution over actions.

This expression shows that entropy regularization is equivalent to adding a state-action dependent penalty term $-\alpha \ln \pi(a|s)$ to the reward. Intuitively, this term amounts to paying a cost for low-entropy (deterministic) policies.

Now, when we write down the Bellman equation for this entropy-regularized problem, at each state $s$ we need to find the decision rule $d(\cdot|s) \in \Delta(\mathcal{A}_s)$ (a probability distribution over actions) that maximizes:

$$
v(s) = \max_{d(\cdot|s) \in \Delta(\mathcal{A}_s)} \sum_a d(a|s) \left[ r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j) - \alpha \ln d(a|s) \right].
$$

Here $\Delta(\mathcal{A}_s) = \{d(\cdot|s) : d(a|s) \geq 0, \sum_a d(a|s) = 1\}$ denotes the probability simplex over actions available at state $s$. The optimization is over randomized decision rules at each state, constrained to be valid probability distributions.

This is a convex optimization problem with a linear constraint. We form the Lagrangian:

$$
\mathcal{L}(d, \lambda) = \sum_a d(a|s) \left[ r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j) - \alpha \ln d(a|s) \right] - \lambda \left(\sum_a d(a|s) - 1\right),
$$
where $\lambda$ is the Lagrange multiplier enforcing the normalization constraint. Taking the derivative with respect to $d(a|s)$ and setting it to zero:

$$
r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j) - \alpha(1 + \ln d^*(a|s)) - \lambda = 0.
$$

Solving for $d^*(a|s)$:

$$
d^*(a|s) = \exp\left(\frac{1}{\alpha}\left(r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j) - \lambda\right)\right).
$$

Using the normalization constraint $\sum_a d^*(a|s) = 1$ to solve for $\lambda$:

$$
\sum_a \exp\left(\frac{1}{\alpha}\left(r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right)\right) = \exp\left(\frac{\lambda}{\alpha}\right).
$$

Therefore:

$$
\lambda = \alpha \ln \sum_a \exp\left(\frac{1}{\alpha}\left(r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right)\right).
$$

Substituting this back into the Bellman equation and simplifying:

$$
v(s) = \alpha \ln \sum_a \exp\left(\frac{1}{\alpha}\left(r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right)\right).
$$

Setting $\beta = 1/\alpha$ (the inverse temperature), this becomes:

$$
v(s) = \frac{1}{\beta} \ln \sum_a \exp\left(\beta\left(r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right)\right).
$$

We recover the smooth Bellman equation we derived earlier using the logsumexp operator. The inverse temperature parameter $\beta$ controls how closely the logsumexp approximates the max: as $\beta \to \infty$, we recover the standard Bellman equation, while for finite $\beta$, we have a smooth approximation that corresponds to optimizing with entropy regularization strength $\alpha = 1/\beta$.

The optimal policy is:

$$
\pi^*(a|s) = \frac{\exp\left(\beta q^*(s,a)\right)}{\sum_{a'} \exp\left(\beta q^*(s,a')\right)} = \text{softmax}_\beta(q^*(s,\cdot))(a),
$$
which is exactly the softmax policy parametrized by inverse temperature.

The derivation establishes the complete equivalence: the value function $v^*$ that solves the smooth Bellman equation is identical to the optimal value function $v^*_\Omega$ of the entropy-regularized MDP (with $\Omega$ being negative entropy and $\alpha = 1/\beta$), and the softmax policy that is greedy with respect to this value function achieves the maximum of the entropy-regularized objective. Both approaches yield the same numerical solution: the same values at every state and the same policy prescriptions. The only difference is how we conceptualize the problem: as smoothing the Bellman operator for computational tractability, or as explicitly trading off reward maximization against policy entropy.

This equivalence has important implications. When we use smooth Bellman equations with a logsumexp operator, we are implicitly solving an entropy-regularized MDP. Conversely, when we explicitly add entropy regularization to an MDP objective, we arrive at smooth Bellman equations as the natural description of optimality. This dual perspective will prove valuable in understanding various algorithms and theoretical results. For instance, in soft actor-critic methods and other maximum entropy reinforcement learning algorithms, the connection between smooth operators and entropy regularization provides both computational benefits (differentiability) and conceptual clarity (why we want stochastic policies).

### Entropy-Regularized Dynamic Programming Algorithms

While the smooth Bellman equations (using logsumexp) and entropy-regularized formulations are mathematically equivalent, it is instructive to present the algorithms explicitly in the entropy-regularized form, where the entropy bonus appears directly in the update equations.

```{prf:algorithm} Entropy-Regularized Value Iteration
:label: entropy-regularized-value-iteration

**Input:** MDP $(S, A, r, p, \gamma)$, entropy weight $\alpha > 0$, tolerance $\epsilon > 0$

**Output:** Approximate optimal value function $v$ and stochastic policy $\pi$

1. Initialize $\pi(a|s) \leftarrow 1/|A_s|$ for all $s \in S, a \in A_s$ (uniform policy)
2. Initialize $v(s) \leftarrow 0$ for all $s \in S$
3. **repeat**
4. $\quad \Delta \leftarrow 0$
5. $\quad$ **for** each state $s \in S$ **do**
6. $\quad\quad$ **Policy Improvement:** Update policy for current value estimate
7. $\quad\quad$ **for** each action $a \in A_s$ **do**
8. $\quad\quad\quad q(s,a) \leftarrow r(s,a) + \gamma \sum_{j \in S} p(j|s,a) v(j)$
9. $\quad\quad$ **end for**
10. $\quad\quad$ **for** each action $a \in A_s$ **do**
11. $\quad\quad\quad \pi_{\text{new}}(a|s) \leftarrow \frac{\exp(q(s,a)/\alpha)}{\sum_{a' \in A_s} \exp(q(s,a')/\alpha)}$
12. $\quad\quad$ **end for**
13. $\quad\quad$ **Value Update:** Compute regularized value
14. $\quad\quad v_{\text{new}}(s) \leftarrow \sum_{a \in A_s} \pi_{\text{new}}(a|s) \cdot q(s,a) + \alpha H(\pi_{\text{new}}(\cdot|s))$
15. $\quad\quad$ where $H(\pi_{\text{new}}(\cdot|s)) = -\sum_{a \in A_s} \pi_{\text{new}}(a|s) \log \pi_{\text{new}}(a|s)$
16. $\quad\quad \Delta \leftarrow \max(\Delta, |v_{\text{new}}(s) - v(s)|)$
17. $\quad\quad v(s) \leftarrow v_{\text{new}}(s)$
18. $\quad\quad \pi(\cdot|s) \leftarrow \pi_{\text{new}}(\cdot|s)$
19. $\quad$ **end for**
20. **until** $\Delta < \epsilon$
21. **return** $v, \pi$
```

**Features:**
- Line 11 updates the policy using the softmax of Q-values, with temperature $\alpha$
- Line 14 explicitly computes the entropy-regularized value: expected Q-value plus entropy bonus
- The algorithm maintains and updates a stochastic policy throughout
- As $\alpha \to 0$ (or equivalently $\beta \to \infty$), this recovers standard value iteration

```{prf:algorithm} Entropy-Regularized Policy Iteration
:label: policy-iteration-entropy-regularized

**Input:** MDP $(S, A, r, p, \gamma)$, entropy weight $\alpha > 0$, tolerance $\epsilon > 0$

**Output:** Approximate optimal value function $v$ and stochastic policy $\pi$

1. Initialize $\pi(a|s) \leftarrow 1/|A_s|$ for all $s \in S, a \in A_s$ (uniform policy)
2. **repeat**
3. $\quad$ **Policy Evaluation:** Solve for $v^\pi$ such that for all $s \in S$:
4. $\quad\quad$ **Option 1 (Iterative):**
5. $\quad\quad$ Initialize $v(s) \leftarrow 0$ for all $s \in S$
6. $\quad\quad$ **repeat**
7. $\quad\quad\quad$ **for** each state $s \in S$ **do**
8. $\quad\quad\quad\quad$ Compute $q^\pi(s,a) \leftarrow r(s,a) + \gamma \sum_{j \in S} p(j|s,a) v(j)$ for all $a \in A_s$
9. $\quad\quad\quad\quad v_{\text{new}}(s) \leftarrow \sum_{a \in A_s} \pi(a|s) \cdot q^\pi(s,a) + \alpha H(\pi(\cdot|s))$
10. $\quad\quad\quad$ **end for**
11. $\quad\quad\quad$ **if** $\max_s |v_{\text{new}}(s) - v(s)| < \epsilon$ **then break**
12. $\quad\quad\quad v \leftarrow v_{\text{new}}$
13. $\quad\quad$ **until** convergence
14. $\quad\quad$ **Option 2 (Direct):** Solve linear system $(\mathbf{I} - \gamma \mathbf{P}_\pi) \mathbf{v} = \mathbf{r}_\pi + \alpha \mathbf{H}_\pi$
15. $\quad\quad$ where $[\mathbf{r}_\pi](s) = \sum_a \pi(a|s) r(s,a)$ and $[\mathbf{H}_\pi](s) = H(\pi(\cdot|s))$
16. $\quad$ **Policy Improvement:**
17. $\quad$ policy_changed $\leftarrow$ false
18. $\quad$ **for** each state $s \in S$ **do**
19. $\quad\quad \pi_{\text{old}}(\cdot|s) \leftarrow \pi(\cdot|s)$
20. $\quad\quad$ **for** each action $a \in A_s$ **do**
21. $\quad\quad\quad q(s,a) \leftarrow r(s,a) + \gamma \sum_{j \in S} p(j|s,a) v(j)$
22. $\quad\quad$ **end for**
23. $\quad\quad$ **for** each action $a \in A_s$ **do**
24. $\quad\quad\quad \pi(a|s) \leftarrow \frac{\exp(q(s,a)/\alpha)}{\sum_{a' \in A_s} \exp(q(s,a')/\alpha)}$
25. $\quad\quad$ **end for**
26. $\quad\quad$ **if** $\|\pi(\cdot|s) - \pi_{\text{old}}(\cdot|s)\| > \epsilon$ **then**
27. $\quad\quad\quad$ policy_changed $\leftarrow$ true
28. $\quad\quad$ **end if**
29. $\quad$ **end for**
30. **until** policy_changed $=$ false
31. **return** $v, \pi$
```

**Features:**
- **Policy Evaluation** (lines 3-15): Computes the value of the current policy including entropy bonus
  - Option 1: Iterative method (successive approximation)
  - Option 2: Direct solution via linear system
- **Policy Improvement** (lines 16-29): Updates policy to softmax over Q-values
- Line 14 shows the vector form: the linear system includes the entropy vector $\mathbf{H}_\pi$
- The algorithm alternates between evaluating the current stochastic policy and improving it
- Converges to the unique optimal entropy-regularized policy

(sec-rust-inverse)=
## Rust's Bus Engines and Inverse Reinforcement Learning

What can we learn about a decision maker's goals from records of what they did?

Rust's 1987 paper is worth reading for three reasons: it contains the smooth Bellman equation, it poses what is now called inverse reinforcement learning, and it solves that problem with a bilevel method that relies on implicit differentiation, a tool that became popular in machine learning three decades later for hyperparameter optimization.

**The problem.** Harold Zurcher was the superintendent of maintenance at the Madison Metropolitan bus company. Rust had his maintenance records: for each bus, the odometer reading each month, and the months in which Zurcher replaced the engine. Replacing an engine is expensive, while keeping an old engine running costs more and more as the mileage grows. The question was how Zurcher decided when to replace. Rust is a structural econometrician. Structural econometrics does not stop at predicting behavior; it posits a mechanism that generates it, here a decision maker solving an MDP, and fits the parameters of that mechanism to data. The fitted mechanism can then answer questions the data alone cannot, such as how behavior would change if the price of an engine changed.

**The model.** The state $s$ is the mileage since the last replacement, discretized into bins. Each month there are two actions, keep ($a=0$) or replace ($a=1$). Keeping costs $c(s;\theta)$, increasing in $s$; replacing costs a fixed amount $RC$ and resets the mileage. With parameters $\theta$ collecting $RC$ and those of $c$, the reward is $r_\theta(s,0) = -c(s;\theta)$ and $r_\theta(s,1) = -RC - c(0;\theta)$, and the mileage transitions are estimated directly from the odometer data. The data contradict any deterministic policy: two buses with the same mileage are sometimes treated differently. Rust's answer was the Gumbel model of {ref}`sec-gumbel-rewards`. Zurcher sees, in addition to $s$, a shock $\epsilon(a)$ for each action that the econometrician does not see, and acts optimally given both. Integrating out the shocks gives the smooth Bellman equation and the softmax choice probabilities

$$
v_\theta = \mathrm{L}_{\beta,\theta}\, v_\theta, \qquad \pi_\theta(a \mid s) = \frac{\exp(\beta q_\theta(s,a))}{\sum_{a'}\exp(\beta q_\theta(s,a'))}.
$$

This is the second, deeper reason Rust used these equations. Besides being smooth, they follow from the random-utility models of {cite:t}`McFadden1974`, the standard statistical basis of discrete choice in econometrics, in the same way that least squares follows from assuming Gaussian noise. In machine learning terms, McFadden's conditional logit is multiclass logistic regression with a structural interpretation.

**The estimator.** Given observed pairs $(s_t, a_t)$, maximize the log-likelihood $\ell(\theta) = \sum_t \log \pi_\theta(a_t \mid s_t)$. Every evaluation of $\ell$ requires solving the smooth Bellman equation for the current $\theta$, so the problem is bilevel: an outer maximization over $\theta$ around an inner fixed-point problem. Rust called his algorithm the nested fixed point (NFXP) algorithm. The gradient of $\ell$ needs $\partial v_\theta / \partial\theta$, which the implicit function theorem provides without differentiating through the iterations of the inner solver. Differentiating $v_\theta = \mathrm{L}_{\beta,\theta} v_\theta$ and using the derivative of $\mathrm{L}_\beta$ from {ref}`sec-smooth-pi-newton`,

$$
\frac{\partial v_\theta}{\partial \theta} = (\mathbf{I} - \gamma\mathbf{P}_{\pi_\theta})^{-1}\, \frac{\partial \mathbf{r}_{\pi_\theta}}{\partial\theta}, \qquad \frac{\partial \mathbf{r}_{\pi_\theta}}{\partial\theta}(s) = \sum_a \pi_\theta(a\mid s)\, \frac{\partial r_\theta(s,a)}{\partial \theta}.
$$

The matrix to invert is the one of policy evaluation, already factored by the Newton steps of the inner solver. For rewards linear in the parameters, $r_\theta(s,a) = \theta^\top\phi(s,a)$, this derivative is the vector of expected discounted feature counts under the softmax policy, the quantity that {cite:t}`ziebart2008maximum` matches to the empirical feature counts. The score of one observation is then

$$
\nabla_\theta \log\pi_\theta(a\mid s) = \beta\Big(\nabla_\theta q_\theta(s,a) - \sum_{a'}\pi_\theta(a'\mid s)\,\nabla_\theta q_\theta(s,a')\Big), \qquad \nabla_\theta q_\theta(s,a) = \nabla_\theta r_\theta(s,a) + \gamma\sum_j p(j\mid s,a)\,\frac{\partial v_\theta}{\partial \theta}(j).
$$

````{prf:algorithm} Nested Fixed Point (NFXP) Maximum Likelihood
:label: nfxp

**Input:** Observed state-action pairs $\{(s_t, a_t)\}$, transition model $p$, discount $\gamma$, scale $\beta$, initial parameters $\theta_0$  
**Output:** Estimated reward parameters $\hat\theta$

1. **repeat** for $k = 0, 1, \ldots$
    1. (Inner problem) Solve $v = \mathrm{L}_{\beta,\theta_k} v$: a few sweeps of {prf:ref}`smooth-value-iteration`, then Newton steps ({ref}`sec-smooth-pi-newton`), warm-started from the previous $v$
    2. Compute $\pi_{\theta_k}$, $q_{\theta_k}$ and $\partial v_{\theta_k}/\partial\theta$ by the implicit-function formula above
    3. Compute $\nabla \ell(\theta_k) = \sum_t \nabla_\theta\log\pi_{\theta_k}(a_t\mid s_t)$
    4. (Outer step) $\theta_{k+1} \leftarrow \theta_k + H_k^{-1}\nabla\ell(\theta_k)$, with $H_k$ a positive definite curvature estimate (Rust used the outer product of the scores, the BHHH method)
2. **until** $\|\nabla\ell(\theta_k)\|$ is small
3. **return** $\hat\theta = \theta_k$
````

**What the model assumes.** The setup is a strong one, and naming its assumptions explains why inverse RL is hard in general.

- *Rationality.* The decision maker is assumed to solve the MDP exactly. People do not; the literature on bounded rationality studies how they fall short, and Rust himself has written skeptically about whether people follow Bellman's principle of optimality. The assumption is still a useful starting point, in the same way a linear model is.
- *Observability.* Rust observes the state and the action. In many applications one or both are missing. With logged prices, for instance, an unchanged price may be a decision to keep it or simply the absence of a decision, and the action itself must be inferred, which makes the problem partially observable.
- *Identifiability.* Many reward functions explain the same behavior equally well. With hard maxima, a constant reward makes every policy optimal, and adding a shaping term $\gamma\Phi(s') - \Phi(s)$ to any reward leaves the optimal policies unchanged {cite}`NgHaradaRussell1999`. The map from rewards to behavior is many-to-one, so the data can at best single out a set of candidate rewards. Structural econometrics restores identification by restricting the reward to a parametric family, fixing the discount factor and the scale $\beta$, and normalizing one action's reward. Whether such restrictions are acceptable depends on the goal: predicting what Zurcher would do under new engine prices is a weaker demand than recovering what he "really" wanted.

## Summary and Outlook

Entropy regularization replaces hard action selection with a smooth conjugate
operator and produces a stochastic optimal policy. Gumbel perturbations,
log-sum-exp Bellman updates, and regularized policy iteration describe the same
construction from probabilistic, variational, and algorithmic viewpoints.
Smooth policy iteration is Newton's method on the smooth Bellman equation, and
Rust's nested fixed-point estimator uses the same derivative to recover reward
parameters from observed behavior.

The same log-sum-exp form appears over whole trajectories in continuous time.
[Path-integral stochastic control](path-integral-stochastic-control.md) examines
systems driven by Gaussian disturbances whose Hamilton--Jacobi--Bellman equation
becomes linear under a logarithmic transformation, so that optimal feedback can
be estimated from weighted trajectories.

Smoothness changes the operator but does not remove the need to represent its
unknown value function. How can an infinite-dimensional functional equation be
reduced to finitely many coefficients and test conditions? [Weighted-residual
methods](weighted-residual-methods.md) provide the general approximation
machinery.

## Self-checks

:::{exercise} Temperature limit
:label: ex-smoothing-check-1

What happens to a softmax policy as the entropy temperature $\alpha$ approaches zero, assuming a unique maximizing action?
:::

:::{solution} ex-smoothing-check-1
:class: dropdown

Its probability concentrates on the maximizing action, so the soft policy approaches the ordinary greedy deterministic policy.
:::

:::{exercise} Smooth maximum
:label: ex-smoothing-check-2

Why is log-sum-exp often easier to optimize through than a hard maximum?
:::

:::{solution} ex-smoothing-check-2
:class: dropdown

Log-sum-exp is differentiable and blends information from all actions. A hard maximum is nondifferentiable at ties and passes local gradient information only through selected maximizers.
:::
