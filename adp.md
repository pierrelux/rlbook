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

# Approximate Dynamic Programming

Dynamic programming methods suffer from the curse of dimensionality and can quickly become difficult to apply in practice. Not only this, we may also be dealing with large or continuous state or action spaces. We have seen so far that we could address this problem using discretization, or interpolation. These were already examples of approximate dynamic programming. In this chapter, we will see other forms of approximations meant to facilitate the optimization problem, either by approximating the optimality equations, the value function, or the policy itself.
Approximation theory is at the heart of learning methods, and fundamentally, this chapter will be about the application of learning ideas to solve complex decision-making problems.

# Smooth Optimality Equations for Infinite-Horizon MDPs

While the standard Bellman optimality equations use the max operator to determine the best action, an alternative formulation known as the smooth or soft Bellman optimality equations replaces this with a softmax operator. This approach originated from {cite}`rust1987optimal` and was later rediscovered in the context of maximum entropy inverse reinforcement learning {cite}`ziebart2008maximum`, which then led to soft Q-learning {cite}`haarnoja2017reinforcement` and soft actor-critic {cite}`haarnoja2018soft`, a state-of-the-art deep reinforcement learning algorithm.

In the infinite-horizon setting, the smooth Bellman optimality equations take the form:

$$ v_\gamma^\star(s) = \frac{1}{\beta} \log \sum_{a \in A_s} \exp\left(\beta\left(r(s, a) + \gamma \sum_{j \in S} p(j | s, a) v_\gamma^\star(j)\right)\right) $$

Adopting an operator-theoretic perspective, we can define a nonlinear operator $\mathrm{L}_\beta$ such that the smooth value function of an MDP is then the solution to the following fixed-point equation:

$$ (\mathrm{L}_\beta \mathbf{v})(s) = \frac{1}{\beta} \log \sum_{a \in \mathcal{A}_s} \exp\left(\beta\left(r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right)\right) $$

As $\beta \to \infty$, $\mathrm{L}_\beta$ converges to the standard Bellman operator $\mathrm{L}$. Furthermore, it can be shown that the smooth Bellman operator is a contraction mapping in the supremum norm, and therefore has a unique fixed point. However, as opposed to the usual "hard" setting, the fixed point of $\mathrm{L}_\beta$ is associated with the value function of an optimal stochastic policy defined by the softmax distribution:

   $$ d(a|s) = \frac{\exp\left(\beta\left(r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v_\gamma^\star(j)\right)\right)}{\sum_{a' \in \mathcal{A}_s} \exp\left(\beta\left(r(s,a') + \gamma \sum_{j \in \mathcal{S}} p(j|s,a') v_\gamma^\star(j)\right)\right)} $$

Despite the confusing terminology, the above "softmax" policy is simply the smooth counterpart to the argmax operator in the original optimality equation: it acts as a soft-argmax. 

This formulation is interesting for several reasons. First, smoothness is a desirable property from an optimization standpoint. Unlike $\gamma$, we view $\beta$ as a hyperparameter of our algorithm, which we can control to achieve the desired level of accuracy.

Second, while presented from an intuitive standpoint where we replace the max by the log-sum-exp (a smooth maximum) and the argmax by the softmax (a smooth argmax), this formulation can also be obtained from various other perspectives, offering theoretical tools and solution methods. For example, {cite:t}`rust1987optimal` derived this algorithm by considering a setting in which the rewards are stochastic and perturbed by a Gumbel noise variable. When considering the corresponding augmented state space and integrating the noise, we obtain smooth equations. This interpretation is leveraged by Rust for modeling purposes.

There is also a way to obtain this equation by starting from the energy-based formulation often used in supervised learning, in which we convert an unnormalized probability distribution into a distribution using the softmax transformation. This is essentially what {cite:t}`ziebart2008maximum` did in their paper. Furthermore, this perspective bridges with the literature on probabilistic graphical models, in which we can now cast the problem of finding an optimal smooth policy into one of maximum likelihood estimation (an inference problem). This is the idea of control as inference, which also admits the converse - that of inference as control - used nowadays for deriving fast samples and amortized inference techniques using reinforcement learning {cite}`levine2018reinforcement`.

Finally, it's worth noting that we can also derive this form by considering an entropy-regularized formulation in which we penalize for the entropy of our policy in the reward function term. This formulation admits a solution that coincides with the smooth Bellman equations {cite}`haarnoja2017reinforcement`.

## Gumbel Noise on the Rewards

We can obtain the smooth Bellman equation by considering a setting in which we have Gumbel noise added to the reward function. More precisely, we define an MDP whose state space is now that of $\tilde{s} = (s, \epsilon)$, where the reward function is given by 

$$\tilde{r}(\tilde{s}, a) = r(s,a) + \epsilon(a)$$

and where the transition probability function is:

$$ p(\tilde{s}' | \tilde{s}, a) = p(s' | s, a) \cdot p(\epsilon') $$

This expression stems from the conditional independence assumption that we make on the noise variable given the state. 

Furthermore, we assume that $\epsilon(a)$ is a random variable following a Gumbel distribution with location 0 and scale $1/\beta$. The Gumbel distribution is a continuous probability distribution used to model the maximum (or minimum) of a number of samples of various distributions. Its probability density function is:

$$ f(x; \mu, \beta) = \frac{1}{\beta}\exp\left(-\left(\frac{x-\mu}{\beta}+\exp\left(-\frac{x-\mu}{\beta}\right)\right)\right) $$

where $\mu$ is the location parameter and $\beta$ is the scale parameter. To generate a Gumbel-distributed random variable, one can use the inverse transform sampling method: and set $ X = \mu - \beta \ln(-\ln(U)) $
where $U$ is a uniform random variable on the interval $(0,1)$.

The Bellman equation in this augmented state space becomes:

$$ v_\gamma^\star(\tilde{s}) = \max_{a \in \mathcal{A}_s} \left\{ \tilde{r}(\tilde{s},a) + \gamma \mathbb{E}_{}\left[v_\gamma^\star(\tilde{s}')\mid \tilde{s}, a\right] \right\} $$

Furthermore, since all we did is to define another MDP, we still have a contraction and an optimal stationary policy $d^\infty = (d, d, ...)$ can be found via the following deterministic Markovian decision rule:

$$
d(\tilde{s})  \in \operatorname{argmax}_{a \in \mathcal{A}_s} \left\{ \tilde{r}(\tilde{s},a) + \gamma \mathbb{E}_{}\left[v_\gamma^\star(\tilde{s}')\mid \tilde{s}, a\right] \right\}
$$

Note how the expectation is now over the next augmented state space and is therefore both over the next state in the original MDP and over the next perturbation. While in the general case there isn't much that we can do to simplify the expression for the expectation over the next state in the MDP, we can however leverage a remarkable property of the Gumbel distribution which allows us to eliminate the $\epsilon$ term in the above and recover the familiar smooth Bellman equation. 

For a set of random variables $X_1, \ldots, X_n$, each following a Gumbel distribution with location parameters $\mu_1, \ldots, \mu_n$ and scale parameter $1/\beta$, extreme value theory tells us that:

$$ \mathbb{E}\left[\max_{i} X_i\right] = \frac{1}{\beta} \log \sum_{i=1}^n \exp(\beta\mu_i) $$

In our case, each $X_i$ corresponds to $r(s,a_i) + \epsilon(a_i) + \gamma \mathbb{E}_{s'}[v(s')]$ for a given action $a_i$. The location parameter $\mu_i$ is $r(s,a_i) + \gamma \mathbb{E}_{s'}[v(s')]$, and the scale parameter is $1/\beta$.

Applying this result to our problem, and taking the expectation over the noise $\epsilon$:

$$ \begin{align*}
v_\gamma^\star(s,\epsilon) &= \max_{a \in \mathcal{A}_s} \left\{ r(s,a) + \epsilon(a) + \gamma \mathbb{E}_{s', \epsilon'}\left[v_\gamma^\star(s',\epsilon')\mid s, \epsilon, a\right] \right\} \\
\mathbb{E}_\epsilon[v_\gamma^\star(s,\epsilon)] &= \mathbb{E}_\epsilon\left[\max_{a \in \mathcal{A}_s} \left\{ r(s,a) + \epsilon(a) + \gamma \mathbb{E}_{s', \epsilon'}\left[v_\gamma^\star(s',\epsilon')\mid s, \epsilon, a\right] \right\}\right] \\
&= \frac{1}{\beta} \log \sum_{a \in \mathcal{A}_s} \exp\left(\beta\left(r(s,a) + \gamma \mathbb{E}_{s', \epsilon'}\left[v_\gamma^\star(s',\epsilon')\mid s, a\right]\right)\right) \\
&= \frac{1}{\beta} \log \sum_{a \in \mathcal{A}_s} \exp\left(\beta\left(r(s,a) + \gamma \mathbb{E}_{s'}\left[\mathbb{E}_{\epsilon'}[v_\gamma^\star(s',\epsilon')]\mid s, a\right]\right)\right)
\end{align*}
$$

If we define $v_\gamma^\star(s) = \mathbb{E}_\epsilon[v_\gamma^\star(s,\epsilon)]$, we obtain the smooth Bellman equation:

$$ v_\gamma^\star(s) = \frac{1}{\beta} \log \sum_{a \in \mathcal{A}_s} \exp\left(\beta\left(r(s,a) + \gamma \mathbb{E}_{s'}\left[v_\gamma^\star(s')\mid s, a\right]\right)\right) $$

This final equation is the smooth Bellman equation, which we derived by introducing Gumbel noise to the reward function and leveraging properties of the Gumbel distribution and extreme value theory.

Now, in the same way that we have been able to simplify and specialize the form of the value function under Gumbel noise, we can also derive an expression for the corresponding optimal policy. To see this, we apply similar steps and start with the optimal decision rule for the augmented MDP:

$$
d(\tilde{s}) \in \operatorname{argmax}_{a \in \mathcal{A}_s} \left\{ \tilde{r}(\tilde{s},a) + \gamma \mathbb{E}_{}\left[v_\gamma^\star(\tilde{s}') \mid \tilde{s}, a\right] \right\}
$$

In order to simplify this expression by taking the expectation over the noise variable, we define an indicator function for the event that action $a$ is in the set of optimal actions:

   $$ I_a(\epsilon) = \begin{cases} 
   1 & \text{if } a \in \operatorname{argmax}_{a' \in \mathcal{A}_s} \left\{ r(s,a') + \epsilon(a') + \gamma \mathbb{E}_{s', \epsilon'}\left[v_\gamma^\star(s',\epsilon')\mid s, a'\right] \right\} \\
   0 & \text{otherwise}
   \end{cases} $$

Note that this definition allows us to recover the original expression since:

$$
\begin{align*}
d(s,\epsilon) &= \left\{a \in \mathcal{A}_s : I_a(\epsilon) = 1\right\} \\
&= \left\{a \in \mathcal{A}_s : a \in \operatorname{argmax}_{a' \in \mathcal{A}_s} \left\{ r(s,a') + \epsilon(a') + \gamma \mathbb{E}_{s', \epsilon'}\left[v_\gamma^\star(s',\epsilon')\mid s, \epsilon,  a'\right] \right\}\right\}
\end{align*}
$$

This set-valued -- but determinisic -- function $d(s,\epsilon)$ gives us the set of optimal actions for a given state $s$ and noise realization $\epsilon$. For simplicity, consider the case where the optimal set of actions at $s$ is a singleton such that taking the expection over the noise variable gives us:
$$ \begin{align*}
\mathbb{E}_\epsilon[I_a(\epsilon)] = \mathbb{P}\left(a \in \operatorname{argmax}_{a' \in \mathcal{A}_s} \left\{ r(s,a') + \epsilon(a') + \gamma \mathbb{E}_{s', \epsilon'}\left[v_\gamma^\star(s',\epsilon')\mid s, \epsilon, a'\right] \right\}\right)
\end{align*} $$

Now, we can leverage a key property of the Gumbel distribution. For a set of random variables $\{X_i = \mu_i + \epsilon_i\}$ where $\epsilon_i$ are i.i.d. Gumbel(0, 1/β) random variables, we have:

   $$ P(X_i \geq X_j, \forall j \neq i) = \frac{\exp(\beta\mu_i)}{\sum_j \exp(\beta\mu_j)} $$

In our case, $X_a = r(s,a) + \epsilon(a) + \gamma \mathbb{E}_{s', \epsilon'}\left[v_\gamma^\star(s',\epsilon')\mid s, a\right]$ for each action $a$, with $\mu_a = r(s,a) + \gamma \mathbb{E}_{s', \epsilon'}\left[v_\gamma^\star(s',\epsilon')\mid s, a\right]$. 

Applying this property and using the definition $v_\gamma^\star(s) = \mathbb{E}_\epsilon[v_\gamma^\star(s,\epsilon)]$, we get:

   $$ d(a|s) = \frac{\exp\left(\beta\left(r(s,a) + \gamma \mathbb{E}_{s'}\left[v_\gamma^\star(s')\mid s, a\right]\right)\right)}{\sum_{a' \in \mathcal{A}_s} \exp\left(\beta\left(r(s,a') + \gamma \mathbb{E}_{s'}\left[v_\gamma^\star(s')\mid s, a'\right]\right)\right)} $$


This gives us the optimal stochastic policy for the smooth MDP. Note that as $\beta \to \infty$, this policy approaches the deterministic policy of the original MDP, while for finite $\beta$, it gives a stochastic policy.

## Control as Inference Perspective

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

$$ \log \beta_t(s_t) = \log \sum_{a_t} \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) \exp(\beta (r(s_t, a_t) + \gamma V(s_{t+1}) + \frac{1}{\beta} \log \beta_{t+1}(s_{t+1}))) $$

If we define the soft value function as $V_t(s_t) = \frac{1}{\beta} \log \beta_t(s_t)$, we can rewrite the above equation as:

$$ V_t(s_t) = \frac{1}{\beta} \log \sum_{a_t} \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) \exp(\beta (r(s_t, a_t) + \gamma V_{t+1}(s_{t+1}))) $$

This is exactly the smooth Bellman equation we derived earlier, but now interpreted as the result of probabilistic inference in a graphical model.

### Deriving the Optimal Policy

The backward message recursion we derived earlier assumes a uniform prior policy $p(a_t | s_t)$. However, our goal is to find not just any policy, but an optimal one. We can extract this optimal policy efficiently by computing the posterior distribution over actions given our backward messages.

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

$$ d(a_t | s_t) = \frac{\exp(\beta (r(s_t, a_t) + \gamma \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) V_{t+1}(s_{t+1})))}{\sum_{a'_t} \exp(\beta (r(s_t, a'_t) + \gamma \sum_{s_{t+1}} p(s_{t+1} | s_t, a'_t) V_{t+1}(s_{t+1})))} $$

## Regularized Markov Decision Processes

Regularized MDPs {cite}`geist2019` provide another perspective on how the smooth Bellman equations come to be. This framework offers a more general approach in which we seek to find optimal policies under the infinite horizon criterion while also accounting for a regularizer that influences the kind of policies we try to obtain.

Let's set up some necessary notation. First, recall that the policy evaluation operator for a stationary policy with decision rule $d$ is defined as:

$$ L_d v = r_d + \gamma P_d v $$

where $r_d$ is the expected reward under policy $d$, $\gamma$ is the discount factor, and $P_d$ is the state transition probability matrix under $d$. A complementary object to the value function is the q-function (or Q-factor) representation:

$$ \begin{align*}
q_\gamma^{d^\infty}(s, a) &= r(s, a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v_\gamma^{d^\infty}(j) \\
v_\gamma^{d^\infty}(s) &= \sum_{a \in \mathcal{A}_s} d(a | s) q_\gamma^{d^\infty}(s, a) 
\end{align*} $$

The policy evaluation operator can then be written in terms of the q-function as:

$$ [L_d v](s) = \langle d(\cdot | s), q(s, \cdot) \rangle $$

### Legendre-Fenchel Transform

A key concept in the theory of regularized MDPs is the Legendre-Fenchel transform, also known as the convex conjugate. For a strongly convex function $\Omega: \Delta_{\mathcal{A}} \rightarrow \mathbb{R}$, its Legendre-Fenchel transform $\Omega^*: \mathbb{R}^{\mathcal{A}} \rightarrow \mathbb{R}$ is defined as:

$$ \Omega^*(q(s, \cdot)) = \max_{d(\cdot|s) \in \Delta_{\mathcal{A}}} \langle d(\cdot | s), q(s, \cdot) \rangle - \Omega(d(\cdot | s)) $$

An important property of this transform is that it has a unique maximizing argument, given by the gradient of $\Omega^*$. This gradient is Lipschitz and satisfies:

$$ \nabla \Omega^*(q(s, \cdot)) = \arg\max_d \langle d(\cdot | s), q(s, \cdot) \rangle - \Omega(d(\cdot | s)) $$

An important example of a regularizer is the negative entropy, which gives rise to the smooth Bellman equations as we are about to see. 

## Regularized Bellman Operators

With these concepts in place, we can now define the regularized Bellman operators:

1. **Regularized Policy Evaluation Operator** $(L_{d,\Omega})$:

   $$ [L_{d,\Omega} v](s) = \langle q(s,\cdot), d(\cdot | s) \rangle - \Omega(d(\cdot | s)) $$

2. **Regularized Bellman Optimality Operator** $(L_\Omega)$:
           
   $$ [L_\Omega v](s) = [\max_d L_{d,\Omega} v ](s) = \Omega^*(q(s, \cdot)) $$

It can be shown that the addition of a regularizer in these regularized operators still preserves the contraction properties, and therefore the existence of a solution to the optimality equations and the convergence of successive approximation.

The regularized value function of a stationary policy with decision rule $d$, denoted by $v_{d,\Omega}$, is the unique fixed point of the operator equation:

$$\text{find $v$ such that } \enspace v = L_{d,\Omega} v$$

Under the usual assumptions on the discount factor and the boundedness of the reward, the value of a policy can also be found in closed form by solving for $v$ in the linear system of equations:

$$ (I - \gamma P_d) v =  (r_d - \Omega(d)) $$

The associated state-action value function $q_{d,\Omega}$ is given by:

$$\begin{align*}
q_{d,\Omega}(s, a) &= r(s, a) + \sum_{j \in \mathcal{S}} \gamma p(j|s,a) v_{d,\Omega}(j) \\
v_{d,\Omega}(s) &= \sum_{a \in \mathcal{A}_s} d(a | s) q_{d,\Omega}(s, a) - \Omega(d(\cdot | s))
\end{align*} $$

The regularized optimal value function $v^*_\Omega$ is then the unique fixed point of $L_\Omega$ in the fixed point equation:

$$\text{find $v$ such that } v = L_\Omega v$$

The associated state-action value function $q^*_\Omega$ is given by:

$$ \begin{align*}
q^*_\Omega(s, a) &= r(s, a) + \sum_{j \in \mathcal{S}} \gamma p(j|s,a) v^*_\Omega(j) \\
v^*_\Omega(s) &= \Omega^*(q^*_\Omega(s, \cdot))\end{align*} $$

An important result in the theory of regularized MDPs is that there exists a unique optimal regularized policy. Specifically, if $d^*_\Omega$ is a conserving decision rule (i.e., $d^*_\Omega = \arg\max_d L_{d,\Omega} v^*_\Omega$), then the randomized stationary policy $(d^*_\Omega)^\infty$ is the unique optimal regularized policy.

In practice, once we have found $v^*_\Omega$, we can derive the optimal decision rule by taking the gradient of the convex conjugate evaluated at the optimal action-value function:

$$ d^*(\cdot | s) = \nabla \Omega^*(q^*_\Omega(s, \cdot)) $$

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

This is precisely the form of the smooth Bellman equation we derived earlier, with the log-sum-exp operation replacing the max operation of the standard Bellman equation.

Furthermore, the optimal policy is given by the gradient of $\Omega^*$:

$$ d^*(a|s) = \nabla \Omega^*(q^*_\Omega(s, \cdot)) = \frac{\exp(q^*_\Omega(s,a))}{\sum_{a' \in \mathcal{A}_s} \exp(q^*_\Omega(s,a'))} $$

This is the familiar softmax policy we encountered in the smooth MDP setting.

# Parametric Dynamic Programming

We have so far considered a specific kind of approximation: that of the Bellman operator itself. We explored a modified version of the operator with the desirable property of smoothness, which we deemed beneficial for optimization purposes and due to its rich multifaceted interpretations. We now turn our attention to another form of approximation, complementary to the previous kind, which seeks to address the challenge of applying the operator across the entire state space.

To be precise, suppose we can compute the Bellman operator $\mathrm{L}v$ at some state $s$, producing a new function $U$ whose value at state $s$ is $u(s) = (\mathrm{L}v)(s)$. Then, putting aside the problem of pointwise evaluation, we want to carry out this update across the entire domain of $v$. When working with small state spaces, this is not an issue, and we can afford to carry out the update across the entirety of the state space. However, for larger or infinite state spaces, this becomes a major challenge.

So what can we do? Our approach will be to compute the operator at chosen "grid points," then "fill in the blanks" for the states where we haven't carried out the update by "fitting" the resulting output function on a dataset of input-output pairs. The intuition is that for sufficiently well-behaved functions and sufficiently expressive function approximators, we hope to generalize well enough. Our community calls this "learning," while others would call it "function approximation" — a field of its own in mathematics. To truly have a "learning algorithm," we'll need to add one more piece of machinery: the use of samples — of simulation — to pick the grid points and perform numerical integration. But this is for the next section...

## Partial Updates in the Tabular Case

The ideas presented in this section apply more broadly to the successive approximation method applied to a fixed-point problem. Consider again the problem of finding the optimal value function $v_\gamma^\star$ as the solution to the Bellman optimality operator $L$: 

$$
\mathrm{L} \mathbf{v} \equiv \max_{d \in D^{MD}} \left\{\mathbf{r}_d + \gamma \mathbf{P}_d \mathbf{v}\right\}
$$

Value iteration — the name for the method of successive approximation applied to $L$ — computes a sequence of iterates $v_{n+1} = \mathrm{L}v_n$ from some arbitrary $v_0$. Let's pause to consider what the equality sign in this expression means: it represents an assignment (perhaps better denoted as $:=$) across the entire domain. This becomes clearer when writing the update in component form:

$$
v_{n+1}(s) := (\mathrm{L} v_n)(s) \equiv \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v_n(j)\right\}, \, \forall s \in \mathcal{S}
$$

Pay particular attention to the $\forall s \in \mathcal{S}$ notation: what happens when we can't afford to update all components in each step of value iteration? A potential solution is to use Gauss-Seidel Value Iteration, which updates states sequentially, immediately using fresh values for subsequent updates. 

````{prf:algorithm} Gauss-Seidel Value Iteration
:label: alg-gsvi

**Input:** MDP $(S, A, P, R, \gamma)$, convergence threshold $\varepsilon > 0$  
**Output:** Value function $v$ and policy $d$

1. **Initialization:**
   - Initialize $v^0(s)$ for all $s \in S$
   - Set iteration counter $n = 0$

2. **Main Loop:**
   - Set state index $j = 1$
   
   a) **State Update:** Compute $v^{n+1}(s_j)$ as:

      $$
      v^{n+1}(s_j) = \max_{a \in A_j} \left\{r(s_j, a) + \gamma \left[\sum_{i<j} p(s_i|s_j,a)v^{n+1}(s_i) + \sum_{i \geq j} p(s_i|s_j,a)v^n(s_i)\right]\right\}
      $$
   
   b) If $j = |S|$, proceed to step 3
      Otherwise, increment $j$ and return to step 2(a)

3. **Convergence Check:**
   - If $\|v^{n+1} - v^n\| < \varepsilon(1-\gamma)/(2\gamma)$, proceed to step 4
   - Otherwise, increment $n$ and return to step 2

4. **Policy Extraction:**
   For each $s \in S$, compute optimal policy:

   $$
   d(s) \in \operatorname{argmax}_{a \in A_s} \left\{r(s,a) + \gamma\sum_{j \in S} p(j|s,a)v^{n+1}(j)\right\}
   $$

**Note:** The algorithm differs from standard value iteration in that it immediately uses updated values within each iteration. This is reflected in the first sum of step 2(a), where $v^{n+1}$ is used for already-updated states.
````

The Gauss-Seidel value iteration approach offers several advantages over standard value iteration: it can be more memory-efficient and often leads to faster convergence. This idea generalizes further (see for example {cite:t}`Bertsekas1983`) to accommodate fully asynchronous updates in any order. However, these methods, while more flexible in their update patterns, still fundamentally rely on a tabular representation—that is, they require storing and eventually updating a separate value for each state in memory. Even if we update states one at a time or in blocks, we must maintain this complete table of values, and our convergence guarantee assumes that every entry in this table will eventually be revised.

But what if maintaining such a table is impossible? This challenge arises naturally when dealing with continuous state spaces, where we cannot feasibly store values for every possible state, let alone update them. This is where function approximation comes into play. 

## Partial Updates by Operator Fitting: Parametric Value Iteration

In the parametric approach to dynamic programming, instead of maintaining an explicit table of values, we represent the value function using a parametric function approximator $v(s; \boldsymbol{\theta})$, where $\boldsymbol{\theta}$ are parameters that get adjusted across iterations rather than the entries of a tabular representation. This idea traces back to the inception of dynamic programming and was described as early as 1963 by Bellman himself, who considered polynomial approximations. For a value function $v(s)$, we can write its polynomial approximation as:

$$
v(s) \approx \sum_{i=0}^{n} \theta_i \phi_i(s)
$$

where:
- $\{\phi_i(s)\}$ is the set of basis functions
- $\theta_i$ are the coefficients (our parameters)
- $n$ is the degree of approximation

As we discussed earlier in the context of trajectory optimization, we can choose from different polynomial bases beyond the usual monomial basis $\phi_i(s) = s^i$, such as Legendre or Chebyshev polynomials. While polynomials offer attractive mathematical properties, they become challenging to work with in higher dimensions due to the curse of dimensionality. This limitation motivates our later turn to neural network parameterizations, which scale better with dimensionality.

Given a parameterization, our value iteration procedure must now update the parameters $\boldsymbol{\theta}$ rather than tabular values directly. At each iteration, we aim to find parameters that best approximate the Bellman operator's output at chosen base points. More precisely, we collect a dataset:

$$
\mathcal{D}_n = \{(s_i, (\mathrm{L}v)(s_i; \boldsymbol{\theta}_n)) \mid s_i \in B\}
$$

and fit a regressor $v(\cdot; \boldsymbol{\theta}_{n+1})$ to this data.

This process differs from standard supervised learning in a specific way: rather than working with a fixed dataset, we iteratively generate our training targets using the previous value function approximation. During this process, the parameters $\boldsymbol{\theta}_n$ remain "frozen", entering only through dataset creation. This naturally leads to maintaining two sets of parameters:
- $\boldsymbol{\theta}_n$: parameters of the target model used for generating training targets
- $\boldsymbol{\theta}_{n+1}$: parameters being optimized in the current iteration

This target model framework emerges naturally from the structure of parametric value iteration — an insight that provides theoretical grounding for modern deep reinforcement learning algorithms where we commonly hear about the importance of the "target network trick" .

Parametric successive approximation, known in reinforcement learning literature as Fitted Value Iteration, offers a flexible template for deriving new algorithms by varying the choice of function approximator. Various instantiations of this approach have emerged across different fields:

- Using polynomial basis functions with linear regression yields Kortum's method {cite}`kortum1992value`, known to econometricians. In reinforcement learning terms, this corresponds to value iteration with projected Bellman equations {cite}`Rust1996`.

- Employing extremely randomized trees (via `ExtraTreesRegressor`) leads to the tree-based fitted value iteration of Ernst et al. {cite}`ernst2005tree`.

- Neural network approximation (via `MLPRegressor`) gives rise to Neural Fitted Q-Iteration as developed by Riedmiller {cite}`riedmiller2005neural`.

The \texttt{fit} function in our algorithm represents this supervised learning step and can be implemented using any standard regression tool that follows the scikit-learn interface. This flexibility in choice of function approximator allows practitioners to leverage the extensive ecosystem of modern machine learning tools while maintaining the core dynamic programming structure.

````{prf:algorithm} Parametric Value Iteration
:label: parametric-value-iteration

**Input** Given an MDP $(S, A, P, R, \gamma)$, base points $B \subset S$, function approximator class $v(s; \boldsymbol{\theta})$, maximum iterations $N$, tolerance $\varepsilon > 0$

**Output** Parameters $\boldsymbol{\theta}$ for value function approximation

1. Initialize $\boldsymbol{\theta}_0$ (e.g., for zero initialization)
2. $n \leftarrow 0$
3. **repeat**

    1. $\mathcal{D} \leftarrow \emptyset$
    2. For each $s \in B$:
        1. $y_s \leftarrow \max_{a \in A} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)v(j; \boldsymbol{\theta}_n)\right\}$
        2. $\mathcal{D} \leftarrow \mathcal{D} \cup \{(s, y_s)\}$

    3. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D})$
    4. $\delta \leftarrow \frac{1}{|B|}\sum_{s \in B} (v(s; \boldsymbol{\theta}_{n+1}) - v(s; \boldsymbol{\theta}_n))^2$
    5. $n \leftarrow n + 1$

4. **until** ($\delta < \varepsilon$ or $n \geq N$)
5. **return** $\boldsymbol{\theta}_n$
````
The structure of the above algorithm mirrors value iteration in its core idea of iteratively applying the Bellman operator. However, several key modifications distinguish this fitted variant:

First, rather than applying updates across the entire state space, we compute the operator only at selected base points $B$. The resulting values are then stored implicitly through the parameter vector $\boldsymbol{\theta}$ via the fitting step, rather than explicitly as in the tabular case.

The fitting procedure itself may introduce an "inner optimization loop." For instance, when using neural networks, this involves an iterative gradient descent procedure to optimize the parameters. This creates an interesting parallel with modified policy iteration: just as we might truncate policy evaluation steps there, we can consider variants where this inner loop runs for a fixed number of iterations rather than to convergence.

Finally, the termination criterion from standard value iteration may no longer hold. The classical criterion relied on the sup-norm contractivity property of the Bellman operator — a property that isn't generally preserved under function approximation. While certain function approximation schemes can maintain this sup-norm contraction property (as we'll see later), this is the exception rather than the rule.

## Parametric Policy Iteration

We can extend this idea of fitting partial operator updates to the policy iteration setting. Remember, policy iteration involves iterating in the space of policies rather than in the space of value functions. Given an initial guess on a deterministic decision rule $d_0$, we iteratively:
1. Compute the value function for the current policy (policy evaluation)
2. Derive a new improved policy (policy improvement)

When computationally feasible and under the model-based setting, we can solve the policy evaluation step directly as a linear system equation. Alternatively, we could carry out policy evaluation by applying successive approximation to the operator $L_{d_n}$ until convergence, or as in modified policy iteration, for just a few steps.

To apply the idea of fitting partial updates, we start at the level of the policy evaluation operator $L_{d_n}$. For a given decision rule $d_n$, this operator in component form is:

$$
(L_{d_n}v)(s) = r(s,d_n(s)) + \gamma \int v(s')p(ds'|s,d_n(s))
$$

For a set of base points $B = \{s_1, ..., s_M\}$, we form our dataset:

$$
\mathcal{D}_n = \{(s_k, y_k) : s_k \in B\}
$$

where:

$$
y_k = r(s_k,d_n(s_k)) + \gamma \int v_n(s')p(ds'|s_k,d_n(s_k))
$$

This gives us a way to perform approximate policy evaluation through function fitting. However, we now face the question of how to perform policy improvement in this parametric setting. A key insight comes from the the fact that in the exact form of policy iteration, we don't need to improve the policy everywhere to guarantee progress. In fact, improving the policy at even a single state is sufficient for convergence.

This suggests a natural approach: rather than trying to approximate an improved policy over the entire state space, we can simply:

1. Compute improved actions at our base points:

$$
d_{n+1}(s_k) = \arg\max_{a \in \mathcal{A}} \left\{r(s_k,a) + \gamma \int v_n(s')p(ds'|s_k,a)\right\}, \quad \forall s_k \in B
$$

2. Let the function approximation of the value function implicitly generalize these improvements to other states during the next policy evaluation phase.

This leads to the following algorithm:

````{prf:algorithm} Parametric Policy Iteration
:label: parametric-policy-iteration

**Input** Given an MDP $(S, A, P, R, \gamma)$, base points $B \subset S$, function approximator class $v(s; \boldsymbol{\theta})$, maximum iterations $N$, tolerance $\varepsilon > 0$

**Output** Parameters $\boldsymbol{\theta}$ for value function approximation

1. Initialize $\boldsymbol{\theta}_0$, decision rules $\{d_0(s_k)\}_{s_k \in B}$
2. $n \leftarrow 0$
3. **repeat**
    1. // Policy Evaluation
    2. $\mathcal{D} \leftarrow \emptyset$
    3. For each $s_k \in B$:
        1. $y_k \leftarrow r(s_k,d_n(s_k)) + \gamma \int v_n(s')p(ds'|s_k,d_n(s_k))$
        2. $\mathcal{D} \leftarrow \mathcal{D} \cup \{(s_k, y_k)\}$
    4. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D})$
    
    5. // Policy Improvement at Base Points
    6. For each $s_k \in B$:
        1. $d_{n+1}(s_k) \leftarrow \arg\max_{a \in A} \{r(s_k,a) + \gamma \int v_n(s')p(ds'|s_k,a)\}$
    
    7. $n \leftarrow n + 1$
4. **until** ($n \geq N$ or convergence criterion met)
5. **return** $\boldsymbol{\theta}_n$
````

As opposed to exact policy iteration, the iterates of parametric policy iteration need not converge monotonically to the optimal value function. Intuitively, this is because we use function approximation to generalize  from base points to the entire state space which can lead to Value estimates improving at base points but degrading at other states or can cause interference between updates at different states due to the shared parametric representation

# Approximate Component-wise Evaluation of the Bellman Operator

So far, we've described a strategy that economizes computation by updating only a selected set of base points, then generalizing these updates to approximate the operator's effect on other states. However, this approach assumes we can compute the operator exactly at these base points. What if even this limited computation proves infeasible?

The challenge arises naturally when we consider what computing the Bellman operator entails: evaluating an expectation over next states, which generally requires numerical integration. Even in the "best" scenario — the model-based setting we've worked with throughout this course — we assume access to an explicit representation of the transition probability function that we can evaluate everywhere. 

But what if we lack these exact probabilities and instead only have access to samples of next states for given state-action pairs? In this case, we must turn to Monte Carlo integration methods, which brings us fully into what we would recognize as a learning setting.

## Numerical Quadrature Methods

In the general case, the Bellman operator in component-wise form is:

$$
(\mathrm{L} v_n)(s) \equiv \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \int v_n(s')p(ds'|s,a)\right\}, \, \forall s \in \mathcal{S}
$$

When we have direct access to $p$ (rather than just samples from it), we can employ numerical integration methods to approximate the expectation. Let's divide our state space into $m$ equal intervals of width $\Delta s = (s_{\max} - s_{\min})/m$. Then, a quadrature approximation takes the form:

$$
\int v_n(s')p(ds'|s,a) \approx \sum_{i=1}^m w_i v_n(s'_i)
$$

For instance, using the rectangle (midpoint) rule with points $s'_i = s_{\min} + (i-\frac{1}{2})\Delta s$:

$$
\int v_n(s')p(ds'|s,a) \approx \sum_{i=1}^m v_n(s'_i)p(s'_i|s,a)\Delta s
$$

or the trapezoidal rule using endpoints $s'_i = s_{\min} + i\Delta s$:

$$
\int v_n(s')p(ds'|s,a) \approx \frac{\Delta s}{2}\sum_{i=1}^{m} [v_n(s'_i)p(s'_i|s,a) + v_n(s'_{i+1})p(s'_{i+1}|s,a)]
$$

It's instructive to contrast this approach with the one that we would apply in the deterministic case:

$$
(\mathrm{L}v)(s) = \max_{a \in \mathcal{A}} \{r(s,a) + \gamma v(f(s,a))\}
$$

where $f(s,a)$ is the deterministic transition function: a special case where $p(ds'|s,a)$ is a Dirac delta measure concentrated at $f(s,a)$. 
This setting does not require numerical integration but instead faces the challenge of evaluating $v$ at $f(s,a)$, which might not coincide with our grid points. This is a problem which we handled through interpolation in the previous chapter by:

1. Maintaining values at grid points $\{s_i\}_{i=1}^N$
2. Using interpolation to evaluate $v$ at arbitrary points

For example with linear interpolation between grid points, $L$ took the form:

$$
(\mathrm{L}v)(s_i) = \max_{a \in \mathcal{A}} \left\{r(s_i,a) + \gamma \left[(1-\alpha)v(s_k) + \alpha v(s_{k+1})\right]\right\}
$$

where $f(s_i,a)$ falls between grid points $s_k$ and $s_{k+1}$, and $\alpha$ is the interpolation weight:

$$
\alpha = \frac{f(s_i,a) - s_k}{s_{k+1} - s_k}
$$

Returning to the Bellman operator in the stochastic case, we can approximate the expectation using quadrature methods. Plugging such a quadrature approximation back into our operator, we obtain $\hat{\mathrm{L}}$:

$$
(\hat{\mathrm{L}} v_n)(s) \equiv \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{i=1}^m v_n(s'_i)p(s'_i|s,a)\Delta s\right\}
$$

This strategy of approximating the operator at the level of expectation computation is complementary to our earlier approach of performing partial updates and then generalizing through function approximation. These approximations address different challenges:
- Quadrature handles the computation of expectations over continuous state spaces (not needed in deterministic case)
- Function approximation deals with representing and generalizing value functions (analogous to interpolation in deterministic case)

Combining both strategies leads to the following algorithm:

````{prf:algorithm} Fitted Value Iteration with Rectangle Rule
:label: fitted-value-iteration-quadrature

**Input** Given an MDP $(S, A, P, R, \gamma)$, base points $B \subset S$, integration points $\{s'_i\}_{i=1}^m$, function approximator class $v(s; \boldsymbol{\theta})$, maximum iterations $N$, tolerance $\varepsilon > 0$

**Output** Parameters $\boldsymbol{\theta}$ for value function approximation

1. Initialize $\boldsymbol{\theta}_0$ (e.g., for zero initialization)
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D} \leftarrow \emptyset$
    2. For each $s \in B$:
        1. $y_s \leftarrow \max_{a \in A} \left\{r(s,a) + \gamma \sum_{i=1}^m v_n(s'_i)p(s'_i|s,a)\Delta s\right\}$
        2. $\mathcal{D} \leftarrow \mathcal{D} \cup \{(s, y_s)\}$
    3. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D})$
    4. $\delta \leftarrow \frac{1}{|B|}\sum_{s \in B} (v(s; \boldsymbol{\theta}_{n+1}) - v(s; \boldsymbol{\theta}_n))^2$
    5. $n \leftarrow n + 1$
4. **until** ($\delta < \varepsilon$ or $n \geq N$)
5. **return** $\boldsymbol{\theta}_n$
````

## Monte Carlo Integration 

Numerical quadrature methods scale poorly with increasing dimension. Specifically, for a fixed error tolerance $\epsilon$, the number of required quadrature points grows exponentially with dimension $d$ as $O((\frac{1}{\epsilon})^d)$. Furthermore, quadrature methods require explicit evaluation of the transition probability function $p(s'|s,a)$ at specified points—a luxury we don't have in the "model-free" setting where we only have access to samples from the MDP.

Let $\mathcal{B} = \{s_1, ..., s_M\}$ be our set of base points where we will evaluate the operator. At each base point $s_k \in \mathcal{B}$, Monte Carlo integration approximates the expectation using $N$ samples:

$$
\int v_n(s')p(ds'|s_k,a) \approx \frac{1}{N} \sum_{i=1}^N v_n(s'_{k,i}), \quad s'_{k,i} \sim p(\cdot|s_k,a)
$$

where $s'_{k,i}$ denotes the $i$-th sample drawn from $p(\cdot|s_k,a)$ for base point $s_k$.

This approach has two remarkable properties:
1. The convergence rate is $O(\frac{1}{\sqrt{N}})$ regardless of the number of dimensions
2. It only requires samples from $p(\cdot|s_k,a)$, not explicit probability values

These properties make Monte Carlo integration particularly attractive for high-dimensional problems and model-free settings. Indeed, this is one of the key mathematical foundations that enables learning in general: the ability to estimate expectations using only samples from a distribution. This principle underlies the empirical risk minimization framework in statistical learning theory, where we approximate expected losses using finite samples.

The resulting approximate Bellman operator at each base point becomes:

$$
(\hat{\mathrm{L}} v_n)(s_k) \equiv \max_{a \in \mathcal{A}_{s_k}} \left\{r(s_k,a) + \frac{\gamma}{N} \sum_{i=1}^N v_n(s'_{k,i})\right\}, \quad s'_{k,i} \sim p(\cdot|s_k,a)
$$

for each $s_k \in \mathcal{B}$. From these $M$ point-wise evaluations, we will then fit our function approximator, just as we did in the quadrature case.

# Q-Factor Representation

As we discussed above, Monte Carlo integration is the method of choice when it comes to approximating the effect of the Bellman operator. This is due to both its computational advantages in higher dimensions and its compatibility with the model-free assumption. However, there is an additional important detail that we have neglected to properly cover: extracting actions from values in a model-free fashion. While we can obtain a value function using the Monte Carlo approach described above, we still face the challenge of extracting an optimal policy from this value function.

More precisely, recall that an optimal decision rule takes the form:

$$
d(s) = \arg\max_{a \in \mathcal{A}} \left\{r(s,a) + \gamma \int v(s')p(ds'|s,a)\right\}
$$

Therefore, even given an optimal value function $v$, deriving an optimal policy would still require Monte Carlo integration every time we query the decision rule/policy at a state.

An important idea in dynamic programming is that rather than approximating a state-value function, we can instead approximate a state-action value function. These two functions are related: the value function is the expectation of the Q-function (called Q-factors by some authors in the operations research literature) over the conditional distribution of actions given the current state:

$$
v(s) = \mathbb{E}[q(s,a)|s]
$$

If $q^*$ is an optimal state-action value function, then $v^*(s) = \max_a q^*(s,a)$. Just as we had a Bellman operator for value functions, we can also define an optimality operator for Q-functions. In component form:

$$
(\mathrm{L}q)(s,a) = r(s,a) + \gamma \int p(ds'|s,a)\max_{a' \in \mathcal{A}(s')} q(s', a')
$$

Furthermore, this operator for Q-functions is also a contraction in the sup-norm and therefore has a unique fixed point $q^*$.

The advantage of iterating over Q-functions rather than value functions is that we can immediately extract optimal actions without having to represent the reward function or transition dynamics directly, nor perform numerical integration. Indeed, an optimal decision rule at state $s$ is obtained as:

$$
d(s) = \arg\max_{a \in \mathcal{A}(s)} q(s,a)
$$

With this insight, we can adapt our parametric value iteration algorithm to work with Q-functions:

````{prf:algorithm} Parametric Q-Value Iteration
:label: parametric-q-value-iteration

**Input** Given an MDP $(S, A, P, R, \gamma)$, base points $B \subset S$, actions $A$, function approximator class $q(s,a; \boldsymbol{\theta})$, maximum iterations $N$, tolerance $\varepsilon > 0$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D} \leftarrow \emptyset$
    2. For each $(s,a) \in B \times A$:
        1. Draw $N$ samples: $s'_i \sim p(\cdot|s,a)$ for $i=1,\ldots,N$
        2. $y_{s,a} \leftarrow r(s,a) + \frac{\gamma}{N} \sum_{i=1}^N \max_{a' \in A} q(s'_i,a'; \boldsymbol{\theta}_n)$
        3. $\mathcal{D} \leftarrow \mathcal{D} \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D})$
    4. $\delta \leftarrow \frac{1}{|B||A|}\sum_{(s,a) \in B \times A} (q(s,a; \boldsymbol{\theta}_{n+1}) - q(s,a; \boldsymbol{\theta}_n))^2$
    5. $n \leftarrow n + 1$
4. **until** ($\delta < \varepsilon$ or $n \geq N$)
5. **return** $\boldsymbol{\theta}_n$
````