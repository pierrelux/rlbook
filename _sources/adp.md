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

$$ \log \beta_t(s_t) = \log \sum_{a_t} \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) \exp(\beta (r(s_t, a_t) + \gamma v(_{t+1}) + \frac{1}{\beta} \log \beta_{t+1}(s_{t+1}))) $$

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

# Approximating the Bellman Operator using Numerical Integration

For both the "hard" and smooth Bellman operators, evaluating the expectation over next states can be challenging when the state space is very large or continuous. In scenarios where we have access to an explicit representation of the transition probability function—the so-called "model-based" setting we've worked with throughout this course—this problem can be addressed using numerical integration methods. But what if we lack these exact probabilities and instead only have access to samples of next states for given state-action pairs? In this case, we must turn to Monte Carlo integration methods, which brings us fully into what we would recognize as a learning setting.

## Discretization and Numerical Quadrature

Recall that the Bellman operator for continuous state spaces takes the form:

$$ (\mathrm{L} v)(s) \equiv \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \int v(s')p(ds'|s,a)\right\}, \, \forall s \in \mathcal{S} $$

To make this computationally tractable for continuous state spaces, we can discretize the state space and approximate the integral over this discretized space. While this allows us to evaluate the Bellman operator componentwise, we must first decide how to represent value functions in this discretized setting.

When working with discretized representations, we partition the state space into $N_s$ cells with centers at grid points $\{s_i\}_{i=1}^{N_s}$. We then work with value functions that are piecewise constant on each cell: ie. for any $s \in S$: $v(s) = v(s_{k(s)})$ where $k(s)$ is the index of the cell containing $s$. We denote the discretized reward function by 
$ r_h(s, a) \equiv r(s_{k(s)}, a) $.

For transition probabilities, we need to be more careful. While we similarly map any state-action pair to its corresponding cell, we must ensure that integrating over the discretized transition function yields a valid probability distribution. We achieve this by normalizing:

$$ p_h(s'|s, a) \equiv \frac{p(s_{k(s')}|s_{k(s)}, a)}{\int p(s_{k(s')}|s_{k(s)}, a) ds'} $$


After defining our discretized reward and transition probability functions, we can write down our discretized Bellman operator.
We start with the Bellman operator using our discretized functions $r_h$ and $p_h$. While these functions map to grid points, they're still defined over continuous spaces - we haven't yet dealt with the computational challenge of the integral. With this discretization approach, the value function is piecewise constant over cells. This lets us express the integral as a sum over cells, where each cell's contribution is the probability of transitioning to that cell multiplied by the value at that cell's grid point:
$$ \begin{aligned}
(\widehat{\mathrm{L}}_h v)(s) &= \max_{k=1,\ldots,N_a} \left\{r_h(s, a_k) + \gamma \int v(s')p_h(s'|s, a_k)ds'\right\} \\
&= \max_{k=1,\ldots,N_a} \left\{r_h(s, a_k) + \gamma \int v(s_{k(s')})p_h(s'|s, a_k)ds'\right\} \\
&= \max_{k=1,\ldots,N_a} \left\{r_h(s, a_k) + \gamma \sum_{i=1}^{N_s} v(s_i) \int_{cell_i} p_h(s'|s, a_k)ds'\right\} \\
&= \max_{k=1,\ldots,N_a} \left\{r_h(s, a_k) + \gamma \sum_{i=1}^{N_s} v(s_i)p_h(s_i|s_{k(s)}, a_k)\right\}
\end{aligned} $$

This form makes clear how discretization converts our continuous-space problem into a finite computation: we've replaced integration over continuous space with summation over grid points. The price we pay is that the number of terms in our sum grows exponentially with the dimension of our state space - the familiar curse of dimensionality.

## Monte Carlo Integration 

Numerical quadrature methods scale poorly with increasing dimension. Specifically, for a fixed error tolerance $\epsilon$, the number of required quadrature points grows exponentially with dimension $d$ as $O\left(\left(\frac{1}{\epsilon}\right)^d\right)$. Furthermore, quadrature methods require explicit evaluation of the transition probability function $p(s'|s,a)$ at specified points—a luxury we don't have in the "model-free" setting where we only have access to samples from the MDP.

Let $\mathcal{B} = \{s_1, \ldots, s_M\}$ be our set of base points where we will evaluate the operator. At each base point $s_k \in \mathcal{B}$, Monte Carlo integration approximates the expectation using $N$ samples:

$$
\int v_n(s')p(ds'|s_k,a) \approx \frac{1}{N} \sum_{i=1}^N v_n(s'_{k,i}), \quad s'_{k,i} \sim p(\cdot|s_k,a)
$$

where $s'_{k,i}$ denotes the $i$-th sample drawn from $p(\cdot|s_k,a)$ for base point $s_k$. This approach has two remarkable properties making it particularly attractive for high-dimensional problems and model-free settings:
1. The convergence rate is $O\left(\frac{1}{\sqrt{N}}\right)$ regardless of the number of dimensions
2. It only requires samples from $p(\cdot|s_k,a)$, not explicit probability values

Using this approximation of the expected value over the next state, we can define a new "empirical" Bellman optimality operator:

$$
(\widehat{\mathrm{L}}_N v)(s_k) \equiv \max_{a \in \mathcal{A}_{s_k}} \left\{r(s_k,a) + \frac{\gamma}{N} \sum_{i=1}^N v(s'_{k,i})\right\}, \quad s'_{k,i} \sim p(\cdot|s_k,a)
$$

for each $s_k \in \mathcal{B}$. A direct adaptation of the successive approximation method for this empirical operator leads to:

```{prf:algorithm} Monte Carlo Value Iteration
:label: mc-value-iteration
**Input:** MDP $(S, A, P, R, \gamma)$, , number of samples $N$, tolerance $\varepsilon > 0$, maximum iterations $K$
**Output:** Value function $v$

1. Initialize $v_0(s) = 0$ for all $s \in S$
2. $i \leftarrow 0$
3. **repeat**
    1. For each $s \in \mathcal{B}$:
        1. Draw $s'_{j} \sim p(\cdot|s,a)$ for $j = 1,\ldots,N$
        2. Set $v_{i+1}(s) \leftarrow \max_{a \in A} \left\{r(s,a) + \frac{\gamma}{N} \sum_{j=1}^N v_i(s'_{j})\right\}$
    2. $\delta \leftarrow \|v_{i+1} - v_i\|_{\infty}$
    3. $i \leftarrow i + 1$
4. **until** $\delta < \varepsilon$ or $i \geq K$
5. **return** final $v_{i+1}$
```

Note that the original error bound derived as a termination criterion for value iteration need not hold in this approximate setting. Hence, we use a generic termination criterion based on computational budget and desired tolerance. While this aspect could be improved, we'll focus on a more pressing matter: the algorithm's tendency to produce upwardly biased values. In other words, this algorithm "thinks" the world is more rosy than it actually is - it overestimates values.


### Overestimation Bias in Monte Carlo Value Iteration

In statistics, bias refers to a systematic error where an estimator consistently deviates from the true parameter value. For an estimator $\hat{\theta}$ of a parameter $\theta$, we define bias as: $\text{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta$. While bias isn't always problematic — sometimes we deliberately introduce bias to reduce variance, as in ridge regression — uncontrolled bias can lead to significantly distorted results. In the context of value iteration, this distortion gets amplified even more due to the recursive nature of the algorithm.

Consider how the Bellman operator works in value iteration. At iteration n, we have a value function estimate $v_i(s)$ and aim to improve it by applying the Bellman operator $\mathrm{L}$. The ideal update would be:

$$(\mathrm{{L}}v_i)(s) = \max_{a \in \mathcal{A}(s)} \left\{ r(s,a) + \gamma \int v_i(s') p(ds'|s,a) \right\}$$

However, we can't compute this integral exactly and use Monte Carlo integration instead, drawing $N$ next-state samples for each state and action pair. The bias emerges when we take the maximum over actions:

$$(\widehat{\mathrm{L}}v_i)(s) = \max_{a \in \mathcal{A}(s)} \hat{q}_i(s,a), \enspace \text{where} \enspace \hat{q}_i(s,a) \equiv r(s,a) + \frac{\gamma}{N} \sum_{j=1}^N v_i(s'_j), \quad s'_j \sim p(\cdot|s,a)$$

White the Monte Carlo estimate $\hat{q}_n(s,a)$ is unbiased for any individual action, the empirical Bellman operator is biased upward due to Jensen's inequality, which states that for any convex function $f$, we have $\mathbb{E}[f(X)] \geq f(\mathbb{E}[X])$. Since the maximum operator is convex, this implies:

$$\mathbb{E}[(\widehat{\mathrm{L}}v_i)(s)] = \mathbb{E}\left[\max_{a \in \mathcal{A}(s)} \hat{q}_i(s,a)\right] \geq \max_{a \in \mathcal{A}(s)} \mathbb{E}[\hat{q}_i(s,a)] = (\mathrm{L}v_i)(s)$$

This means that our Monte Carlo approximation of the Bellman operator is biased upward:

$$b_i(s) = \mathbb{E}[(\widehat{\mathrm{L}}v_i)(s)] - (\mathrm{L}v_i)(s) \geq 0$$

Even worse, this bias compounds through iterations as each new value function estimate $v_{n+1}$ is based on targets generated by the biased operator $\widehat{\mathrm{L}}$, creating a nested structure of bias accumulation.
This bias remains nonnegative at every step, and each application of the Bellman operator potentially adds more upward bias. As a result, instead of converging to the true value function $v^*$, the algorithm typically stabilizes at a biased approximation that systematically overestimates true values.

### The Keane-Wolpin Bias Correction Algorithm

Keane and Wolpin proposed to de-bias such estimators by essentially "learning" the bias, then subtracting it when computing the empirical Bellman operator. If we knew this bias function, we could subtract it from our empirical estimate to get an unbiased estimate of the true Bellman operator:

$$
(\widehat{\mathrm{L}}v_n)(s) - \text{bias}(s) = (\widehat{\mathrm{L}}v_n)(s) - (\mathbb{E}[(\widehat{\mathrm{L}}v_n)(s)] - (\mathrm{L}v_n)(s)) \approx (\mathrm{L}v_n)(s)
$$

This equality holds in expectation, though any individual estimate would still have variance around the true value.

So how can we estimate the bias function? The Keane-Wolpin manages this using an important fact from extreme value theory: for normal random variables, the difference between the expected maximum and maximum of expectations scales with the standard deviation:

$$
\mathbb{E}\left[\max_{a \in \mathcal{A}} \hat{q}_i(s,a)\right] - \max_{a \in \mathcal{A}} \mathbb{E}[\hat{q}_i(s,a)] \approx c \cdot \sqrt{\max_{a \in \mathcal{A}} \text{Var}_i(s,a)}
$$

The variance term $\max_{a \in \mathcal{A}} \text{Var}_i(s,a)$ will typically be dominated by the action with the largest value -- the greedy action $a^*_i(s)$. Rather than deriving the constant $c$ theoretically, Keane-Wolpin proposed learning the relationship between variance and bias empirically through these steps:

1. Select a small set of "benchmark" states (typically 20-50) that span the state space
2. For these states, compute more accurate value estimates using many more Monte Carlo samples (10-100x more than usual)
3. Compute the empirical bias at each benchmark state $s$:

   $$\hat{b}_i(s) = (\hat{\mathrm{L}}v_i)(s) - (\hat{\mathrm{L}}_{\text{accurate}}v_i)(s)$$

4. Fit a linear relationship between this bias and the variance at the greedy action:

   $$\hat{b}_i(s) = \alpha_i \cdot \text{Var}_i(s,a^*_i(s)) + \epsilon$$

This creates a dataset of pairs $(\text{Var}_i(s,a^*_i(s)), \hat{b}_i(s))$ that can be used to estimate $\alpha_i$ through ordinary least squares regression. Once we have learned this bias function $\hat{b}$, we can define the bias-corrected Bellman operator:

$$ 
(\widetilde{\mathrm{L}}v_i)(s) \triangleq (\hat{\mathrm{L}}v_i)(s) - \hat{b}(s)
$$

Interestingly, while this bias correction approach has been influential in econometrics, it hasn't gained much traction in the machine learning community. A major drawback is the need for accurate operator estimation at benchmark states, which requires allocating substantially more samples to these states. In the next section, we'll explore an alternative strategy that, while requiring the maintenance of two sets of value estimates, achieves bias correction without demanding additional samples.

### Decoupling Selection and Evaluation

A simpler approach to addressing the upward bias is to maintain two separate q-function estimates - one for action selection and another for evaluation. Let's first start by looking at the corresponding Monte Carlo value iteration algorithm and then convince ourselves that this is good idea using math. Assume a monte carlo integration setup over Q factors: 


```{prf:algorithm} Double Monte Carlo Q-Value Iteration
:label: double-mc-value-iteration

**Input**: MDP $(S, A, P, R, \gamma)$, number of samples $N$, tolerance $\varepsilon > 0$, maximum iterations $K$
**Output**: Q-functions $q^A, q^B$

1. Initialize $q^A_0(s,a) = q^B_0(s,a) = 0$ for all $s \in S, a \in A$
2. $i \leftarrow 0$
3. repeat
    1. For each $s \in S, a \in A$:
        1. Draw $s'_j \sim p(\cdot|s,a)$ for $j = 1,\ldots,N$
        2. For network A:
            1. $a^*_i \leftarrow \arg\max_{a'} q^A_i(s'_j,a')$
            2. $q^A_{i+1}(s,a) \leftarrow r(s,a) + \frac{\gamma}{N} \sum_{j=1}^N q^B_i(s'_j,a^*_i)$
        3. For network B:
            1. $b^*_i \leftarrow \arg\max_{a'} q^B_i(s'_j,a')$
            2. $q^B_{i+1}(s,a) \leftarrow r(s,a) + \frac{\gamma}{N} \sum_{j=1}^N q^A_i(s'_j,b^*_i)$
    2. $\delta \leftarrow \max(\|q^A_{i+1} - q^A_i\|_{\infty}, \|q^B_{i+1} - q^B_i\|_{\infty})$
    3. $i \leftarrow i + 1$
4. until $\delta < \varepsilon$ or $i \geq K$
5. return final $q^A_{i+1}, q^B_{i+1}$
```
In this algorithm, we maintain two separate Q-functions ($q^A$ and $q^B$) and use them asymmetrically: when updating $q^A$, we use network A to select the best action ($a^*_i = \arg\max_{a'} q^A_i(s'_j,a')$) but then evaluate that action using network B's estimates ($q^B_i(s'_j,a^*_i)$). We do the opposite for updating $q^B$. You can see this separation clearly in steps 3.2.2 and 3.2.3 of the algorithm, where for each network update, we first use one network to pick the action and then plug that chosen action into the other network for evaluation. We will see that this decomposition helps mitigate the positive bias that occurs due to Jensen's inequality. 

#### An HVAC analogy

Consider a building where each HVAC unit $i$ has some true maximum power draw $\mu_i$ under worst-case conditions. Let's pretend that we don't have access to manufacturer datasheets, so we need to estimate these maxima from actual measurements. Now the challenge is that power draw fluctuates with environmental conditions. If we use a single day's measurements and look at the highest power draw, we systematically overestimate the true maximum draw across all units. 

To see this, let $X_A^i$ be unit i's power draw on day A and $X_B^i$ be unit i's power draw on day. Wile both measurements are unbiased $\mathbb{E}[X_A^i] = \mathbb{E}[X_B^i] = \mu_i$, their maximum is not due to Jensen's inequality:

$$
\mathbb{E}[\max_i X_A^i] \geq \max_i \mathbb{E}[X_A^i] = \max_i \mu_i
$$

Intuitively, this problem occurs because reading tends to come from units that experienced particularly demanding conditions (e.g., direct sunlight, full occupancy, peak humidity) rather than just those with high true maximum draw. To estimate the true maximum power draw more accurately, we use the following measurement protocol:

1. Use day A measurements to **select** which unit hit the highest peak
2. Use day B measurements to **evaluate** that unit's power consumption

This yields the estimator:

$$
Y = X_B^{\arg\max_i X_A^i}
$$

We can show that by decoupling selection and evaluation in this fashion, our estimator $Y$ will no longer systematically overestimate the true maximum draw. First, observe that $\arg\max_i X_A^i$ is a random variable (call it $J$) - it tells us which unit had highest power draw on day A. It has some probability distribution based on day A's conditions:
   $P(J = j) = P(\arg\max_i X_A^i = j)$.
Using the law of total expectation:

$$
\begin{align*}
\mathbb{E}[Y] = \mathbb{E}[X_B^J] &= \mathbb{E}[\mathbb{E}[X_B^J \mid J]] \text{ (by tower property)} \\
&= \sum_{j=1}^n \mathbb{E}[X_B^j \mid J = j] P(J = j) \\
&= \sum_{j=1}^n \mathbb{E}[X_B^j \mid \arg\max_i X_A^i = j] P(\arg\max_i X_A^i = j)
\end{align*}
$$

Now we need to make an imporant observation: Unit j's power draw on day B ($X_B^j$) is independent of whether it had the highest reading on day A ($\{\arg\max_i X_A^i = j\}$). An extreme cold event on day A shouldn't affect day B's readings(especially in Quebec where the wheather tend to vary widely from day to day). Therefore:

$$
\mathbb{E}[X_B^j \mid \arg\max_i X_A^i = j] = \mathbb{E}[X_B^j] = \mu_j
$$

This tells us that the two-day estimator is now an average of the true underlying power consumptions:

$$
\mathbb{E}[Y] = \sum_{j=1}^n \mu_j P(\arg\max_i X_A^i = j)
$$


To analyze $ \mathbb{E}[Y] $ more closely, let’s use a general result: if we have a real-valued function $ f $ defined on a discrete set of units $ \{1, \dots, n\} $ and a probability distribution $ q(\cdot) $ over these units, then the maximum value of $ f $ across all units is at least as large as the weighted sum of $ f $ values with weights $ q $. Formally,

$$
\max_{j \in \{1, \dots, n\}} f(j) \geq \sum_{j=1}^n q(j) f(j).
$$

Applying this to our setting, we set $ f(j) = \mu_j $ (the true maximum power draw for unit $ j $) and $ q(j) = P(J = j) $ (the probability that unit $ j $ achieves the maximum reading on day A). This gives us:

$$
\max_{j \in \{1, \dots, n\}} \mu_j \geq \sum_{j=1}^n P(J = j) \mu_j = \mathbb{E}[Y].
$$

Therefore, the expected value of $ Y $ (our estimator) will always be less than or equal to the true maximum value $ \max_j \mu_j $. In other words, $ Y $ provides a **conservative estimate** of the true maximum: it tends not to overestimate $ \max_j \mu_j $ but instead approximates it as closely as possible without systematic upward bias.

#### Consistency

Even though $ Y $ is not an unbiased estimator of $ \max_j \mu_j $ (since $ \mathbb{E}[Y] \leq \max_j \mu_j $), it is **consistent**. As more independent days (or measurements) are observed, the selection-evaluation procedure becomes more effective at isolating the intrinsic maximum, reducing the influence of day-specific environmental fluctuations. Over time, this approach yields a stable and increasingly accurate approximation of $ \max_j \mu_j $.

To show that $ Y $ is a consistent estimator of $ \max_i \mu_i $, we want to demonstrate that as the number of independent measurements (days, in this case) increases, $ Y $ converges in probability to $ \max_i \mu_i $. Let's suppose we have $ m $ independent days of measurements for each unit. Denote:
- $ X_A^{(k),i} $ as the power draw for unit $ i $ on day $ A_k $, where $ k \in \{1, \dots, m\} $.
- $ J_m = \arg\max_i \left( \frac{1}{m} \sum_{k=1}^m X_A^{(k),i} \right) $, which identifies the unit with the highest average power draw over $ m $ days.

The estimator we construct is:
$
Y_m = X_B^{(J_m)},
$
where $ X_B^{(J_m)} $ is the power draw of the selected unit $ J_m $ on an independent evaluation day $ B $.
We will now show that $ Y_m $ converges to $ \max_i \mu_i $ as $ m \to \infty $. This involves two main steps:

1. **Consistency of the Selection Step $ J_m $**: As $ m \to \infty $, the unit selected by $ J_m $ will tend to be the one with the true maximum power draw $ \max_i \mu_i $.
2. **Convergence of $ Y_m $ to $ \mu_{J_m} $**: Since the evaluation day $ B $ measurement $ X_B^{(J_m)} $ is unbiased with expectation $ \mu_{J_m} $, as $ m \to \infty $, $ Y_m $ will converge to $ \mu_{J_m} $, which in turn converges to $ \max_i \mu_i $.

The average power draw over $ m $ days for each unit $ i $ is:

$$
\frac{1}{m} \sum_{k=1}^m X_A^{(k),i}.
$$

By the law of large numbers, as $ m \to \infty $, this sample average converges to the true expected power draw $ \mu_i $ for each unit $ i $:

$$
\frac{1}{m} \sum_{k=1}^m X_A^{(k),i} \xrightarrow{m \to \infty} \mu_i.
$$

Since $ J_m $ selects the unit with the highest sample average, in the limit, $ J_m $ will almost surely select the unit with the highest true mean, $ \max_i \mu_i $. Thus, as $ m \to \infty $,
$$
\mu_{J_m} \to \max_i \mu_i.
$$

Given that $ J_m $ identifies the unit with the maximum true mean power draw in the limit, we now look at $ Y_m = X_B^{(J_m)} $, which is the power draw of unit $ J_m $ on the independent evaluation day $ B $.

Since $ X_B^{(J_m)} $ is an unbiased estimator of $ \mu_{J_m} $, we have:

$$
\mathbb{E}[Y_m \mid J_m] = \mu_{J_m}.
$$

As $ m \to \infty $, $ \mu_{J_m} $ converges to $ \max_i \mu_i $. Thus, $ Y_m $ will also converge in probability to $ \max_i \mu_i $ because $ Y_m $ is centered around $ \mu_{J_m} $ and $ J_m $ converges to the index of the unit with $ \max_i \mu_i $.


Combining these two steps, we conclude that:

$$
Y_m \xrightarrow{m \to \infty} \max_i \mu_i \text{ in probability}.
$$

This establishes the **consistency** of $ Y $ as an estimator for $ \max_i \mu_i $: as the number of independent measurements grows, $ Y_m $ converges to the true maximum power draw $ \max_i \mu_i $.

# Parametric Dynamic Programming

We have so far considered a specific kind of approximation: that of the Bellman operator itself. We explored a modified version of the operator with the desirable property of smoothness, which we deemed beneficial for optimization purposes and due to its rich multifaceted interpretations. We now turn our attention to another form of approximation, complementary to the previous kind, which seeks to address the challenge of applying the operator across the entire state space.

To be precise, suppose we can compute the Bellman operator $\mathrm{L}v$ at some state $s$, producing a new function $U$ whose value at state $s$ is $u(s) = (\mathrm{L}v)(s)$. Then, putting aside the problem of pointwise evaluation, we want to carry out this update across the entire domain of $v$. When working with small state spaces, this is not an issue, and we can afford to carry out the update across the entirety of the state space. However, for larger or infinite state spaces, this becomes a major challenge.

So what can we do? Our approach will be to compute the operator at chosen "grid points," then "fill in the blanks" for the states where we haven't carried out the update by "fitting" the resulting output function on a dataset of input-output pairs. The intuition is that for sufficiently well-behaved functions and sufficiently expressive function approximators, we hope to generalize well enough. Our community calls this "learning," while others would call it "function approximation" — a field of its own in mathematics. To truly have a "learning algorithm," we'll need to add one more piece of machinery: the use of samples — of simulation — to pick the grid points and perform numerical integration. But this is for the next section...

## Partial Updates in the Tabular Case

The ideas presented in this section apply more broadly to the successive approximation method applied to a fixed-point problem. Consider again the problem of finding the optimal value function $v_\gamma^\star$ as the solution to the Bellman optimality operator $\mathrm{L}$: 

$$
\mathrm{L} \mathbf{v} \equiv \max_{d \in D^{MD}} \left\{\mathbf{r}_d + \gamma \mathbf{P}_d \mathbf{v}\right\}
$$

Value iteration — the name for the method of successive approximation applied to $\mathrm{L}$ — computes a sequence of iterates $v_{n+1} = \mathrm{L}v_n$ from some arbitrary $v_0$. Let's pause to consider what the equality sign in this expression means: it represents an assignment (perhaps better denoted as $:=$) across the entire domain. This becomes clearer when writing the update in component form:

$$
v_{n+1}(s) := (\mathrm{L} v_n)(s) \equiv \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v_n(j)\right\}, \, \forall s \in \mathcal{S}
$$

Pay particular attention to the $\forall s \in \mathcal{S}$ notation: what happens when we can't afford to update all components in each step of value iteration? A potential solution is to use Gauss-Seidel Value Iteration, which updates states sequentially, immediately using fresh values for subsequent updates. 

```{prf:algorithm} Gauss-Seidel Value Iteration
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
```

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

```{prf:algorithm} Parametric Value Iteration
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
```
The structure of the above algorithm mirrors value iteration in its core idea of iteratively applying the Bellman operator. However, several key modifications distinguish this fitted variant:

First, rather than applying updates across the entire state space, we compute the operator only at selected base points $B$. The resulting values are then stored implicitly through the parameter vector $\boldsymbol{\theta}$ via the fitting step, rather than explicitly as in the tabular case.

The fitting procedure itself may introduce an "inner optimization loop." For instance, when using neural networks, this involves an iterative gradient descent procedure to optimize the parameters. This creates an interesting parallel with modified policy iteration: just as we might truncate policy evaluation steps there, we can consider variants where this inner loop runs for a fixed number of iterations rather than to convergence.

Finally, the termination criterion from standard value iteration may no longer hold. The classical criterion relied on the sup-norm contractivity property of the Bellman operator — a property that isn't generally preserved under function approximation. While certain function approximation schemes can maintain this sup-norm contraction property (as we'll see later), this is the exception rather than the rule.

### Parametric Policy Iteration

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

```{prf:algorithm} Parametric Policy Iteration
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
```

As opposed to exact policy iteration, the iterates of parametric policy iteration need not converge monotonically to the optimal value function. Intuitively, this is because we use function approximation to generalize  from base points to the entire state space which can lead to Value estimates improving at base points but degrading at other states or can cause interference between updates at different states due to the shared parametric representation


## Q-Factor Representation

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

```{prf:algorithm} Parametric Q-Value Iteration
:label: parametric-q-value-iteration

**Input** Given an MDP $(S, A, P, R, \gamma)$, base points $\mathcal{B} \subset S$, function approximator class $q(s,a; \boldsymbol{\theta})$, maximum iterations $N$, tolerance $\varepsilon > 0$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ (e.g., for zero initialization)
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D} \leftarrow \emptyset$
    2. For each $(s,a) \in \mathcal{B} \times A$:
        1. $y_{s,a} \leftarrow r(s,a) + \gamma \int p(ds'|s,a)\max_{a' \in A} q(s',a'; \boldsymbol{\theta}_n)$
        2. $\mathcal{D} \leftarrow \mathcal{D} \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D})$
    4. $\delta \leftarrow \frac{1}{|\mathcal{D}||A|}\sum_{(s,a) \in \mathcal{D} \times A} (q(s,a; \boldsymbol{\theta}_{n+1}) - q(s,a; \boldsymbol{\theta}_n))^2$
    5. $n \leftarrow n + 1$
4. **until** ($\delta < \varepsilon$ or $n \geq N$)
5. **return** $\boldsymbol{\theta}_n$
```

## Warmstarting: The Choice of Initialization

Parametric dynamic programming involves solving a sequence of related optimization problems, one for each fitting procedure at each iteration. While we've presented these as independent fitting problems, in practice we can leverage the relationship between successive iterations through careful initialization. This "warmstarting" strategy can significantly impact both computational efficiency and solution quality.

The basic idea is simple: rather than starting each fitting procedure from scratch, we initialize the function approximator with parameters from the previous iteration. This can speed up convergence since successive Q-functions tend to be similar. However, recent work suggests that persistent warmstarting might sometimes be detrimental, potentially leading to a form of overfitting. Alternative "reset" strategies that occasionally reinitialize parameters have shown promise in mitigating this issue.

Here's how warmstarting can be incorporated into parametric Q-learning with one-step Monte Carlo integration:

```{prf:algorithm} Warmstarted Parametric Q-Learning with N=1 Monte Carlo Integration
:label: warmstarted-q-learning

**Input** Given dataset $\mathcal{D}$ with transitions $(s, a, r, s')$, function approximator class $q(s,a; \boldsymbol{\theta})$, maximum iterations $N$, tolerance $\varepsilon > 0$, warmstart frequency $k$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D} \leftarrow \emptyset$
    2. For each $(s,a,r,s') \in \mathcal{D}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a'} q(s',a'; \boldsymbol{\theta}_n)$  // One-step Monte Carlo estimate
        2. $\mathcal{D} \leftarrow \mathcal{D} \cup \{((s,a), y_{s,a})\}$
    3. **if** $n \bmod k = 0$:  // Reset parameters periodically
        1. Initialize $\boldsymbol{\theta}_{temp}$ randomly
    4. **else**:
        1. $\boldsymbol{\theta}_{temp} \leftarrow \boldsymbol{\theta}_n$  // Warmstart from previous iteration
    5. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}, \boldsymbol{\theta}_{temp})$  // Initialize optimizer with $\boldsymbol{\theta}_{temp}$
    4. $\delta \leftarrow \frac{1}{|\mathcal{D}|}\sum_{(s,a) \in \mathcal{D}} (q(s,a; \boldsymbol{\theta}_{n+1}) - q(s,a; \boldsymbol{\theta}_n))^2$
    5. $n \leftarrow n + 1$
4. **until** ($\delta < \varepsilon$ or $n \geq N$)
5. **return** $\boldsymbol{\theta}_n$
```

The main addition here is the periodic reset of parameters (controlled by frequency $k$) which helps balance the benefits of warmstarting with the need to avoid potential overfitting. When $k=\infty$, we get traditional persistent warmstarting, while $k=1$ corresponds to training from scratch each iteration.

## Inner Optimization: Fit to Convergence or Not?

Beyond the choice of initialization and whether to chain optimization problems through warmstarting, we can also control how we terminate the inner optimization procedure. In the templates presented above, we implicitly assumed that $\texttt{fit}$ is run to convergence. However, this need not be the case, and different implementations handle this differently.

For example, scikit-learn's MLPRegressor terminates based on several criteria: when the improvement in loss falls below a tolerance (default `tol=1e-4`), when it reaches the maximum number of iterations (default `max_iter=200`), or when the loss fails to improve for `n_iter_no_change` consecutive epochs. In contrast, ExtraTreesRegressor builds trees deterministically to completion based on its splitting criteria, with termination controlled by parameters like `min_samples_split` and `max_depth`.

The intuition for using early stopping in the inner optimization mirrors that of modified policy iteration in exact dynamic programming. Just as modified policy iteration truncates the Neumann series during policy evaluation rather than solving to convergence, we might only partially optimize our function approximator at each iteration. While this complicates the theoretical analysis, it often works well in practice and can be computationally more efficient.

This perspective helps us understand modern deep reinforcement learning algorithms. For instance, DQN can be viewed as an instance of fitted Q-iteration where the inner optimization is intentionally limited. Here's how we can formalize this approach:

```{prf:algorithm} Early-Stopping Fitted Q-Iteration
:label: early-stopping-fqi

**Input** Given dataset $\mathcal{D}$ with transitions $(s, a, r, s')$, function approximator $q(s,a; \boldsymbol{\theta})$, maximum outer iterations $N_{outer}$, maximum inner iterations $N_{inner}$, outer tolerance $\varepsilon_{outer}$, inner tolerance $\varepsilon_{inner}$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{P} \leftarrow \emptyset$
    2. For each $(s,a,r,s') \in \mathcal{D}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a'} q(s',a'; \boldsymbol{\theta}_n)$
        2. $\mathcal{P} \leftarrow \mathcal{P} \cup \{((s,a), y_{s,a})\}$
    3. // Inner optimization loop with early stopping
    4. $\boldsymbol{\theta}_{temp} \leftarrow \boldsymbol{\theta}_n$
    5. $k \leftarrow 0$
    6. **repeat**
        1. Update $\boldsymbol{\theta}_{temp}$ using one step of optimizer on $\mathcal{P}$
        2. Compute inner loop loss $\delta_{inner}$
        3. $k \leftarrow k + 1$
    7. **until** ($\delta_{inner} < \varepsilon_{inner}$ or $k \geq N_{inner}$)
    8. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_{temp}$
    9. $\delta_{outer} \leftarrow \frac{1}{|\mathcal{D}|}\sum_{(s,a) \in \mathcal{D}} (q(s,a; \boldsymbol{\theta}_{n+1}) - q(s,a; \boldsymbol{\theta}_n))^2$
    10. $n \leftarrow n + 1$
4. **until** ($\delta_{outer} < \varepsilon_{outer}$ or $n \geq N_{outer}$)
5. **return** $\boldsymbol{\theta}_n$
```
This formulation makes explicit the two-level optimization structure and allows us to control the trade-off between inner loop optimization accuracy and overall computational efficiency. When $N_{inner}=1$, we recover something closer to DQN's update rule, while larger values of $N_{inner}$ bring us closer to the full fitted Q-iteration approach.

# Example Methods

There are several moving parts we can swap in and out when working with parametric dynamic programming - from the function approximator we choose, to how we warm start things, to the specific methods we use for numerical integration and inner optimization. In this section, we'll look at some concrete examples and see how they fit into this general framework.

## Kernel-Based Reinforcement Learning (2002)

Ormoneit and Sen's Kernel-Based Reinforcement Learning (KBRL) {cite}`Ormoneit2002` helped establish the general paradigm of batch reinforcement learning later advocated by {cite}`ErnstGW05`. KBRL is a purely offline method that first collects a fixed set of transitions and then uses kernel regression to solve the optimal control problem through value iteration on this dataset. While the dominant approaches at the time were online methods like temporal difference, KBRL showed that another path to developping reinforcement learning algorithm was possible: one that capable of leveraging advances in supervised learning to provide both theoretical and practical benefits. 

As the name suggests, KBRL uses kernel based regression within the general framework of outlined above. 

```{prf:algorithm} Kernel-Based Q-Value Iteration
:label: kernel-based-q-iteration

**Input** Given an MDP $(S, A, P, R, \gamma)$, dataset $\mathcal{D}$ with observed transitions $(s, a, r, s')$, kernel bandwidth $b$, maximum iterations $N$, tolerance $\varepsilon > 0$

**Output** Kernel-based Q-function approximation

1. Initialize $\hat{Q}_0$ to zero everywhere
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D} \leftarrow \emptyset$
    2. For each $(s, a, r, s') \in \mathcal{D}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} \hat{Q}_n(s', a')$
        2. $\mathcal{D} \leftarrow \mathcal{D} \cup \{((s,a), y_{s,a})\}$
    3. $\hat{Q}_{n+1}(s,a) \leftarrow \sum_{(s_i,a_i,r_i,s_i') \in \mathcal{D}} k_b(s_i, s)\mathbb{1}[a_i=a] y_{s_i,a_i} / \sum_{(s_i,a_i,r_i,s_i') \in \mathcal{D}} k_b(s_i, s)\mathbb{1}[a_i=a]$
    4. $\delta \leftarrow \frac{1}{|\mathcal{D}|}\sum_{(s,a,r,s') \in \mathcal{D}} (\hat{Q}_{n+1}(s,a) - \hat{Q}_n(s,a))^2$
    5. $n \leftarrow n + 1$
4. **until** ($\delta < \varepsilon$ or $n \geq N$)
5. **return** $\hat{Q}_n$
```
Step 3 is where KBRL uses kernel regression with a normalized weighting kernel:

$$k_b(x^l_t, x) = \frac{\phi(\|x^l_t - x\|/b)}{\sum_{l'} \phi(\|x^l_{t'} - x\|/b)}$$

where $\phi$ is a kernel function (often Gaussian) and $b$ is the bandwidth parameter. Each iteration reuses the entire fixed dataset to re-estimate Q-values through this kernel regression.

An important theoretical contribution of KBRL is showing that this kernel-based approach ensures convergence of the Q-function sequence. The authors prove that, with appropriate choice of kernel bandwidth decreasing with sample size, the method is consistent - the estimated Q-function converges to the true Q-function as the number of samples grows.

The main practical limitation of KBRL is computational - being a batch method, it requires storing and using all transitions at each iteration, leading to quadratic complexity in the number of samples. The authors acknowledge this limitation for online settings, suggesting that modifications like discarding old samples or summarizing data clusters would be needed for online applications. Ernst's later work with tree-based methods would help address this limitation while maintaining many of the theoretical advantages of the batch approach.

## Ernst's Fitted Q Iteration (2005)

Ernst's {cite}`ErnstGW05` specific instantiation of parametric q-value iteration uses extremely randomized trees, an extension to random forests proposed by  {cite:t}`Geurts2006`. This algorithm became particularly well-known, partly because it was one of the first to demonstrate the advantages of offline reinforcement learning in practice on several challenging benchmarks at the time. 

Random Forests and Extra-Trees differ primarily in how they construct individual trees. Random Forests creates diversity in two ways: it resamples the training data (bootstrap) for each tree, and at each node it randomly selects a subset of features but then searches exhaustively for the best cut-point within each selected feature. In contrast, Extra-Trees uses the full training set for each tree and injects randomization differently: at each node, it not only randomly selects features but also randomly selects the cut-points without searching for the optimal one. It then picks the best among these completely random splits according to a variance reduction criterion. This double randomization - in both feature and cut-point selection - combined with using the full dataset makes Extra-Trees faster than Random Forests while maintaining similar predictive accuracy.

An important implementation detail concerns how tree structures can be reused across iterations of fitted Q iteration. With parametric methods like neural networks, warmstarting is straightforward - you simply initialize the weights with values from the previous iteration. For decision trees, the situation is more subtle because the model structure is determined by how splits are chosen at each node. When the number of candidate splits per node is $K=1$ (totally randomized trees), the algorithm selects both the splitting variable and threshold purely at random, without looking at the target values (the Q-values we're trying to predict) to evaluate the quality of the split. This means the tree structure only depends on the input variables and random choices, not on what we're predicting. As a result, we can build the trees once in the first iteration and reuse their structure throughout all iterations, only updating the prediction values at the leaves.

Standard Extra-Trees ($K>1$), however, uses target values to choose the best among K random splits by calculating which split best reduces the variance of the predictions. Since these target values change in each iteration of fitted Q iteration (as our estimate of Q evolves), we must rebuild the trees completely. While this is computationally more expensive, it allows the trees to better adapt their structure to capture the evolving Q-function.

The complete algorithm can be formalized as follows:

```{prf:algorithm} Extra-Trees Fitted Q Iteration
:label: extra-trees-fqi

**Input** Given an MDP $(S, A, P, R, \gamma)$, dataset $\mathcal{D}$ with observed transitions $(s, a, r, s')$, Extra-Trees parameters $(K, n_{min}, M)$, maximum iterations $N$, tolerance $\varepsilon > 0$

**Output** Extra-Trees model for Q-function approximation

1. Initialize $\hat{Q}_0$ to zero everywhere
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D} \leftarrow \emptyset$
    2. For each $(s, a, r, s') \in \mathcal{D}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} \hat{Q}_n(s', a')$
        2. $\mathcal{D} \leftarrow \mathcal{D} \cup \{((s,a), y_{s,a})\}$
    3. $\hat{Q}_{n+1} \leftarrow \text{BuildExtraTrees}(\mathcal{D}, K, n_{min}, M)$
    4. $\delta \leftarrow \frac{1}{|\mathcal{D}|}\sum_{(s,a,r,s') \in \mathcal{D}} (\hat{Q}_{n+1}(s,a) - \hat{Q}_n(s,a))^2$
    5. $n \leftarrow n + 1$
4. **until** ($\delta < \varepsilon$ or $n \geq N$)
5. **return** $\hat{Q}_n$
```
## Neural Fitted Q Iteration (2005)

Riedmiller's Neural Fitted Q Iteration (NFQI) {cite}`Riedmiller05` is a natural instantiation of parametric Q-value iteration where:

1. The function approximator $q(s,a; \boldsymbol{\theta})$ is a multi-layer perceptron
2. The $\texttt{fit}$ function uses Rprop optimization trained to convergence on each iteration's pattern set
3. The expected next-state values are estimated through Monte Carlo integration with $N=1$, using the observed next states from transitions

Specifically, rather than using numerical quadrature which would require known transition probabilities, NFQ approximates the expected future value using observed transitions:

$$
\int q_n(s',a')p(ds'|s,a) \approx q_n(s'_{observed},a')
$$

where $s'_{observed}$ is the actual next state that was observed after taking action $a$ in state $s$. This is equivalent to Monte Carlo integration with a single sample, making the algorithm fully model-free.

The algorithm follows from the parametric Q-value iteration template:

```{prf:algorithm} Neural Fitted Q Iteration
:label: neural-fitted-q-iteration

**Input** Given an MDP $(S, A, P, R, \gamma)$, dataset $\mathcal{D}$ with observed transitions $(s, a, r, s')$, MLP architecture $q(s,a; \boldsymbol{\theta})$, maximum iterations $N$, tolerance $\varepsilon > 0$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D} \leftarrow \emptyset$
    2. For each $(s,a,r,s') \in \mathcal{D}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a'} q(s',a'; \boldsymbol{\theta}_n)$  // Monte Carlo estimate with one sample
        2. $\mathcal{D} \leftarrow \mathcal{D} \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{n+1} \leftarrow \text{Rprop}(\mathcal{D})$ // Train MLP to convergence
    4. $\delta \leftarrow \frac{1}{|\mathcal{D}||A|}\sum_{(s,a) \in \mathcal{D} \times A} (q(s,a; \boldsymbol{\theta}_{n+1}) - q(s,a; \boldsymbol{\theta}_n))^2$
    5. $n \leftarrow n + 1$
4. **until** ($\delta < \varepsilon$ or $n \geq N$)
5. **return** $\boldsymbol{\theta}_n$
```

While NFQI was originally introduced as an offline method with base points collected a priori, the authors also present a variant where base points are collected incrementally. In this online variant, new transitions are gathered using the current policy (greedy with respect to $Q_k$) and added to the experience set. This approach proves particularly useful when random exploration cannot efficiently collect representative experiences.

## Deep Q Networks (2013)

DQN {cite}`mnih2013atari` is a close relative of NFQI - in fact, Riedmiller, the author of NFQI, was also an author on the DQN paper. What at first glance might look like a different algorithm can actually be understood as a special case of parametric dynamic programming with practical adaptations. Let's build this connection step by step.

First, let's start with basic parametric Q-value iteration using a neural network:

```{prf:algorithm} Basic Offline Neural Fitted Q-Value Iteration
:label: basic-q-iteration

**Input** Given an MDP $(S, A, P, R, \gamma)$, dataset of transitions $\mathcal{T}$, neural network $q(s,a; \boldsymbol{\theta})$, maximum iterations $N$, tolerance $\varepsilon > 0$, initialization $\boldsymbol{\theta}_0$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $n \leftarrow 0$
3. **repeat**
    1. $\mathcal{D}_n \leftarrow \emptyset$  // Regression dataset
    2. For each $(s,a,r,s') \in \mathcal{T}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_n)$
        2. $\mathcal{D}_n \leftarrow \mathcal{D}_n \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{n+1} \leftarrow \texttt{fit}(\mathcal{D}_n, \boldsymbol{\theta}_0)$ // Fit neural network using built-in convergence criterion 
    4. $n \leftarrow n + 1$
4. **until** training complete
5. **return** $\boldsymbol{\theta}_n$
```
Next, let's open up the `fit` procedure to show the inner optimization loop using gradient descent:

```{prf:algorithm} Fitted Q-Value Iteration with Explicit Inner Loop
:label: q-iteration-inner-loop

**Input** Given MDP $(S, A, P, R, \gamma)$, dataset of transitions $\mathcal{T}$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, convergence test $\texttt{has_converged}(\cdot)$, initialization $\boldsymbol{\theta}_0$, regression loss function $\mathcal{L}$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $n \leftarrow 0$  // Outer iteration index
3. **repeat**
    1. $\mathcal{D}_n \leftarrow \emptyset$  // Regression dataset
    2. For each $(s,a,r,s') \in \mathcal{T}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_n)$
        2. $\mathcal{D}_n \leftarrow \mathcal{D}_n \cup \{((s,a), y_{s,a})\}$
    3. // Inner optimization loop
    4. $\boldsymbol{\theta}^{(0)} \leftarrow \boldsymbol{\theta}_0$  // Start from initial parameters
    5. $k \leftarrow 0$  // Inner iteration index
    6. **repeat**
        1. $\boldsymbol{\theta}^{(k+1)} \leftarrow \boldsymbol{\theta}^{(k)} - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}^{(k)}; \mathcal{D}_n)$
        2. $k \leftarrow k + 1$
    7. **until** $\texttt{has_converged}(\boldsymbol{\theta}^{(0)}, ..., \boldsymbol{\theta}^{(k)}, \mathcal{D}_n)$
    8. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}^{(k)}$
    9. $n \leftarrow n + 1$
4. **until** training complete
5. **return** $\boldsymbol{\theta}_n$
```

### Warmstarting and Partial Fitting

A natural modification is to initialize the inner optimization loop with the previous iteration's parameters - a strategy known as warmstarting - rather than starting from $\boldsymbol{\theta}_0$ each time. Additionally, similar to how modified policy iteration performs partial policy evaluation rather than solving to convergence, we can limit ourselves to a fixed number of optimization steps. These pragmatic changes, when combined, yield:

```{prf:algorithm} Neural Fitted Q-Iteration with Warmstarting and Partial Optimization
:label: nfqi-warmstart-partial

**Input** Given MDP $(S, A, P, R, \gamma)$, dataset of transitions $\mathcal{T}$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, number of steps $K$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $n \leftarrow 0$  // Outer iteration index
3. **repeat**
    1. $\mathcal{D}_n \leftarrow \emptyset$  // Regression dataset
    2. For each $(s,a,r,s') \in \mathcal{T}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_n)$
        2. $\mathcal{D}_n \leftarrow \mathcal{D}_n \cup \{((s,a), y_{s,a})\}$
    3. // Inner optimization loop with warmstart and fixed steps
    4. $\boldsymbol{\theta}^{(0)} \leftarrow \boldsymbol{\theta}_n$  // Warmstart from previous iteration
    5. For $k = 0$ to $K-1$:
        1. $\boldsymbol{\theta}^{(k+1)} \leftarrow \boldsymbol{\theta}^{(k)} - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}^{(k)}; \mathcal{D}_n)$
    6. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}^{(K)}$
    7. $n \leftarrow n + 1$
4. **until** training complete
5. **return** $\boldsymbol{\theta}_n$
```

### Flattening the Updates with Target Swapping

Now rather than maintaining two sets of indices for the outer and inner levels, we could also "flatten" this algorithm  under a single loop structure using modulo arithmetics. Here's how we could rewrite it:

```{prf:algorithm} Flattened Neural Fitted Q-Iteration
:label: nfqi-flattened-swap

**Input** Given MDP $(S, A, P, R, \gamma)$, dataset of transitions $\mathcal{T}$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, target update frequency $K$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_0$  // Initialize target parameters
3. $t \leftarrow 0$  // Single iteration counter
4. **while** training:
    1. $\mathcal{D}_t \leftarrow \emptyset$  // Regression dataset
    2. For each $(s,a,r,s') \in \mathcal{T}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_{target})$  // Use target parameters
        2. $\mathcal{D}_t \leftarrow \mathcal{D}_t \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t; \mathcal{D}_t)$
    4. If $t \bmod K = 0$:  // Every K steps
        1. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_t$  // Update target parameters
    5. $t \leftarrow t + 1$
4. **return** $\boldsymbol{\theta}_t$
```

The flattened version with target parameters achieves exactly the same effect as our previous nested-loop structure with warmstarting and K gradient steps. In the nested version, we would create a dataset using parameters $\boldsymbol{\theta}_n$, then perform K gradient steps to obtain $\boldsymbol{\theta}_{n+1}$. In our flattened version, we maintain a separate $\boldsymbol{\theta}_{target}$ that gets updated every K steps, ensuring that the dataset $\mathcal{D}_n$ is created using the same parameters for K consecutive iterations - just as it would be in the nested version. The only difference is that we've restructured the algorithm to avoid explicitly nesting the loops, making it more suitable for continuous online training which we are about to introduce. The periodic synchronization of $\boldsymbol{\theta}_{target}$ with the current parameters $\boldsymbol{\theta}_n$ effectively marks the boundary of what would have been the outer loop in our previous version.

### Exponential Moving Average Targets

An alternative to this periodic swap of parameters is to use an exponential moving average (EMA) of the parameters:

```{prf:algorithm} Flattened Neural Fitted Q-Iteration with EMA
:label: nfqi-flattened-ema

**Input** Given MDP $(S, A, P, R, \gamma)$, dataset of transitions $\mathcal{T}$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, EMA rate $\tau$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_0$  // Initialize target parameters
3. $n \leftarrow 0$  // Single iteration counter
4. **while** training:
    1. $\mathcal{D}_n \leftarrow \emptyset$  // Regression dataset
    2. For each $(s,a,r,s') \in \mathcal{T}$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_{target})$  // Use target parameters
        2. $\mathcal{D}_n \leftarrow \mathcal{D}_n \cup \{((s,a), y_{s,a})\}$
    3. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_n - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_n; \mathcal{D}_n)$
    4. $\boldsymbol{\theta}_{target} \leftarrow \tau\boldsymbol{\theta}_{n+1} + (1-\tau)\boldsymbol{\theta}_{target}$  // Smooth update of target parameters
    5. $n \leftarrow n + 1$
4. **return** $\boldsymbol{\theta}_n$
```

Note that the original DQN used the periodic swap of parameters rather than EMA targets. 
EMA targets (also called "Polyak averaging") started becoming popular in deep RL with DDPG {cite}`lillicrap2015continuous` where they used a "soft" target update: $\boldsymbol{\theta}_{target} \leftarrow \tau\boldsymbol{\theta} + (1-\tau)\boldsymbol{\theta}_{target}$ with a small $\tau$ (like 0.001). This has since become a common choice in many algorithms like TD3 {cite}`fujimoto2018addressing` and SAC {cite}`haarnoja2018soft`.

### Online Data Collection and Experience Replay

Rather than using offline data, we now consider a modification where we incrementally gather samples under our current policy. A common exploration strategy is $\varepsilon$-greedy: with probability $\varepsilon$ we select a random action, and with probability $1-\varepsilon$ we select the greedy action $\arg\max_a q(s,a;\boldsymbol{\theta}_n)$. This ensures we maintain some exploration even as our Q-function estimates improve. Typically $\varepsilon$ is annealed over time, starting with a high value (e.g., 1.0) to encourage early exploration and gradually decreasing to a small final value (e.g., 0.01) to maintain a minimal level of exploration while mostly exploiting our learned policy.

```{prf:algorithm} Flattened Online Neural Fitted Q-Iteration
:label: online-nfqi-flattened

**Input** Given MDP $(S, A, P, R, \gamma)$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, target update frequency $K$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_0$  // Initialize target parameters
3. Initialize $\mathcal{T} \leftarrow \emptyset$  // Initialize transition dataset
4. $n \leftarrow 0$  // Single iteration counter
5. **while** training:
    1. Observe current state $s$
    2. Select action $a$ using policy derived from $q(s,\cdot;\boldsymbol{\theta}_n)$ (e.g., ε-greedy)
    3. Execute $a$, observe reward $r$ and next state $s'$
    4. $\mathcal{T}_n \leftarrow \mathcal{T}_n \cup \{(s,a,r,s')\}$  // Add transition to dataset
    5. $\mathcal{D}_n \leftarrow \emptyset$  // Regression dataset
    6. For each $(s,a,r,s') \in \mathcal{T}_n$:
        1. $y_{s,a} \leftarrow r + \gamma \max_{a' \in A} q(s',a'; \boldsymbol{\theta}_{target})$  // Use target parameters
        2. $\mathcal{D}_n \leftarrow \mathcal{D}_n \cup \{((s,a), y_{s,a})\}$
    7. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_n - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_n; \mathcal{D}_n)$
    8. If $n \bmod K = 0$:  // Every K steps
        1. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_n$  // Update target parameters
    9. $n \leftarrow n + 1$
6. **return** $\boldsymbol{\theta}_n$
```

This version faces two practical challenges. First, the transition dataset $\mathcal{T}_n$ grows unbounded over time, creating memory issues. Second, computing gradients over the entire dataset becomes increasingly expensive. These are common challenges in online learning settings, and the standard solutions from supervised learning apply here:
1. Use a fixed-size circular buffer (often called replay buffer, in reference to "experience replay" by {cite}`lin1992self`) to limit memory usage
2. Compute gradients on mini-batches rather than the full dataset

Here's how we can modify our algorithm to incorporate these ideas:

```{prf:algorithm} Deep-Q Network 
:label: online-nfqi-replay

**Input** Given MDP $(S, A, P, R, \gamma)$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, target update frequency $K$, replay buffer size $B$, mini-batch size $b$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_0$  // Initialize target parameters
3. Initialize replay buffer $\mathcal{R}$ with capacity $B$
4. $n \leftarrow 0$  // Single iteration counter
5. **while** training:
    1. Observe current state $s$
    2. Select action $a$ using policy derived from $q(s,\cdot;\boldsymbol{\theta}_n)$ (e.g., ε-greedy)
    3. Execute $a$, observe reward $r$ and next state $s'$
    4. Store $(s,a,r,s')$ in $\mathcal{R}$, replacing oldest if full  // Circular buffer update
    5. $\mathcal{D}_n \leftarrow \emptyset$  // Regression dataset
    6. Sample mini-batch of $b$ transitions $(s_i,a_i,r_i,s_i')$ from $\mathcal{R}$
    7. For each sampled $(s_i,a_i,r_i,s_i')$:
        1. $y_i \leftarrow r_i + \gamma \max_{a' \in A} q(s_i',a'; \boldsymbol{\theta}_{target})$  // Use target parameters
        2. $\mathcal{D}_n \leftarrow \mathcal{D}_n \cup \{((s_i,a_i), y_i)\}$
    8. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_n - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_n; \mathcal{D}_n)$ // Replace by RMSProp to obtain DQN
    9. If $n \bmod K = 0$:  // Every K steps
        1. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_n$  // Update target parameters
    10. $n \leftarrow n + 1$
6. **return** $\boldsymbol{\theta}_n$
```

This formulation naturally leads to an important concept in deep reinforcement learning: the replay ratio (or data reuse ratio). In our algorithm, for each new transition we collect, we sample a mini-batch of size b from our replay buffer and perform one update. This means we're reusing past experiences at a ratio of b:1 - for every new piece of data, we're learning from b experiences. This ratio can be tuned as a hyperparameter. Higher ratios mean more computation per environment step but better data efficiency, as we're extracting more learning from each collected transition. This highlights one of the key benefits of experience replay: it allows us to decouple the rate of data collection from the rate of learning updates. Some modern algorithms like SAC or TD3 explicitly tune this ratio, sometimes using multiple gradient steps per environment step to achieve higher data efficiency.

I'll write a subsection that naturally follows from the previous material and introduces double Q-learning in the context of DQN.

### Double-Q Network Variant

As we saw earlier, the max operator in the target computation can lead to overestimation of Q-values. This happens because we use the same network to both select and evaluate actions in the target computation: $y_i \leftarrow r_i + \gamma \max_{a' \in A} q(s_i',a'; \boldsymbol{\theta}_{target})$. The max operator means we're both choosing the action that looks best under our current estimates and then using that same set of estimates to evaluate how good that action is, potentially compounding any optimization bias.

Double DQN {cite:t}`van2016deep` addresses this by using the current network parameters to select actions but the target network parameters to evaluate them. This leads to a simple modification of the DQN algorithm:

```{prf:algorithm} Double Deep-Q Network
:label: double-dqn

**Input** Given MDP $(S, A, P, R, \gamma)$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, target update frequency $K$, replay buffer size $B$, mini-batch size $b$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_0$  // Initialize target parameters
3. Initialize replay buffer $\mathcal{R}$ with capacity $B$
4. $n \leftarrow 0$  // Single iteration counter
5. **while** training:
    1. Observe current state $s$
    2. Select action $a$ using policy derived from $q(s,\cdot;\boldsymbol{\theta}_n)$ (e.g., ε-greedy)
    3. Execute $a$, observe reward $r$ and next state $s'$
    4. Store $(s,a,r,s')$ in $\mathcal{R}$, replacing oldest if full
    5. $\mathcal{D}_n \leftarrow \emptyset$  // Regression dataset
    6. Sample mini-batch of $b$ transitions $(s_i,a_i,r_i,s_i')$ from $\mathcal{R}$
    7. For each sampled $(s_i,a_i,r_i,s_i')$:
        1. $a^*_i \leftarrow \arg\max_{a' \in A} q(s_i',a'; \boldsymbol{\theta}_n)$  // Select action using current network
        2. $y_i \leftarrow r_i + \gamma q(s_i',a^*_i; \boldsymbol{\theta}_{target})$  // Evaluate using target network
        3. $\mathcal{D}_n \leftarrow \mathcal{D}_n \cup \{((s_i,a_i), y_i)\}$
    8. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_n - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_n; \mathcal{D}_n)$
    9. If $n \bmod K = 0$:  // Every K steps
        1. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_n$  // Update target parameters
    10. $n \leftarrow n + 1$
6. **return** $\boldsymbol{\theta}_n$
```

The main difference from the original DQN is in step 7, where we now separate action selection from action evaluation. Rather than directly taking the max over the target network's Q-values, we first select the action using our current network ($\boldsymbol{\theta}_n$) and then evaluate that specific action using the target network ($\boldsymbol{\theta}_{target}$). This simple change has been shown to lead to more stable learning and better final performance across a range of tasks.

## Deep Q Networks with Resets (2022)

In flattening neural fitted Q-iteration, our field had perhaps lost sight of an important structural element: the choice of inner-loop initializer inherent in the original FQI algorithm. The traditional structure explicitly separated outer iterations (computing targets) from inner optimization (fitting to those targets), with each inner optimization starting fresh from parameters $\boldsymbol{\theta}_0$. 

The flattened version with persistent warmstarting seemed like a natural optimization - why throw away learned parameters? However, recent work {cite}`Doro2023` has shown that persistent warmstarting can actually be detrimental to learning. Neural networks tend to lose their ability to learn and generalize over the course of training, suggesting that occasionally starting fresh from $\boldsymbol{\theta}_0$ might be beneficial. Here's how this looks algorithmically in the context of DQN:

```{prf:algorithm} DQN with Hard Resets
:label: dqn-hard-resets

**Input** Given MDP $(S, A, P, R, \gamma)$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, reset interval $K$, replay buffer size $B$, mini-batch size $b$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_0$  // Initialize target parameters
3. Initialize replay buffer $\mathcal{R}$ with capacity $B$
4. $n \leftarrow 0$  // Single iteration counter
5. **while** training:
    1. Observe current state $s$
    2. Select action $a$ using policy derived from $q(s,\cdot;\boldsymbol{\theta}_n)$ (e.g., ε-greedy)
    3. Execute $a$, observe reward $r$ and next state $s'$
    4. Store $(s,a,r,s')$ in $\mathcal{R}$, replacing oldest if full
    5. $\mathcal{D}_n \leftarrow \emptyset$  // Regression dataset
    6. Sample mini-batch of $b$ transitions $(s_i,a_i,r_i,s_i')$ from $\mathcal{R}$
    7. For each sampled $(s_i,a_i,r_i,s_i')$:
        1. $y_i \leftarrow r_i + \gamma \max_{a' \in A} q(s_i',a'; \boldsymbol{\theta}_{target})$
        2. $\mathcal{D}_n \leftarrow \mathcal{D}_n \cup \{((s_i,a_i), y_i)\}$
    8. If $n \bmod K = 0$:  // Periodic reset
        1. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_0$  // Reset to initial parameters
    9. Else:
        1. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_n - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_n; \mathcal{D}_n)$
    10. If $n \bmod K = 0$:  // Every K steps
        1. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_n$  // Update target parameters
    11. $n \leftarrow n + 1$
6. **return** $\boldsymbol{\theta}_n$
```

This algorithm change allows us to push the limits of our update ratio - the number of gradient steps we perform per environment interaction. Without resets, increasing this ratio leads to diminishing returns as the network's ability to learn degrades. However, by periodically resetting the parameters while maintaining our dataset of transitions, we can perform many more updates per interaction, effectively making our algorithm more "offline" and thus more sample efficient. 

The hard reset strategy, while effective, might be too aggressive in some settings as it completely discards learned parameters. An alternative approach is to use a softer form of reset, adapting the "Shrink and Perturb" technique originally introduced by {cite:t}`ash2020warm` in the context of continual learning. In their work, they found that neural networks that had been trained on one task could better adapt to new tasks if their parameters were partially reset - interpolated with a fresh initialization - rather than either kept intact or completely reset.

We can adapt this idea to our setting. Instead of completely resetting to $\boldsymbol{\theta}_0$, we can perform a soft reset by interpolating between our current parameters and a fresh random initialization:

```{prf:algorithm} DQN with Shrink and Perturb
:label: dqn-soft-resets

**Input** Given MDP $(S, A, P, R, \gamma)$, neural network $q(s,a; \boldsymbol{\theta})$, learning rate $\alpha$, reset interval $K$, replay buffer size $B$, mini-batch size $b$, interpolation coefficient $\beta$

**Output** Parameters $\boldsymbol{\theta}$ for Q-function approximation

1. Initialize $\boldsymbol{\theta}_0$ randomly
2. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_0$  // Initialize target parameters
3. Initialize replay buffer $\mathcal{R}$ with capacity $B$
4. $n \leftarrow 0$  // Single iteration counter
5. **while** training:
    1. Observe current state $s$
    2. Select action $a$ using policy derived from $q(s,\cdot;\boldsymbol{\theta}_n)$ (e.g., ε-greedy)
    3. Execute $a$, observe reward $r$ and next state $s'$
    4. Store $(s,a,r,s')$ in $\mathcal{R}$, replacing oldest if full
    5. $\mathcal{D}_n \leftarrow \emptyset$  // Regression dataset
    6. Sample mini-batch of $b$ transitions $(s_i,a_i,r_i,s_i')$ from $\mathcal{R}$
    7. For each sampled $(s_i,a_i,r_i,s_i')$:
        1. $y_i \leftarrow r_i + \gamma \max_{a' \in A} q(s_i',a'; \boldsymbol{\theta}_{target})$
        2. $\mathcal{D}_n \leftarrow \mathcal{D}_n \cup \{((s_i,a_i), y_i)\}$
    8. If $n \bmod K = 0$:  // Periodic soft reset
        1. Sample $\boldsymbol{\phi} \sim$ initializer  // Fresh random parameters
        2. $\boldsymbol{\theta}_{n+1} \leftarrow \beta\boldsymbol{\theta}_n + (1-\beta)\boldsymbol{\phi}$
    9. Else:
        1. $\boldsymbol{\theta}_{n+1} \leftarrow \boldsymbol{\theta}_n - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_n; \mathcal{D}_n)$
    10. If $n \bmod K = 0$:  // Every K steps
        1. $\boldsymbol{\theta}_{target} \leftarrow \boldsymbol{\theta}_n$  // Update target parameters
    11. $n \leftarrow n + 1$
6. **return** $\boldsymbol{\theta}_n$
```

The interpolation coefficient $\beta$ controls how much of the learned parameters we retain, with $\beta = 0$ recovering the hard reset case and $\beta = 1$ corresponding to no reset at all. This provides a more flexible approach to restoring learning capability while potentially preserving useful features that have been learned. Like hard resets, this softer variant still enables high update ratios by preventing the degradation of learning capability, but does so in a more gradual way.

# Does Parametric Dynamic Programming Converge?

So far we have avoided the discussion of convergence and focused on intuitive algorithm development, showing how we can extend successive approximation by computing only a few operator evaluations which then get generalized over the entire domain at each step of the value iteration procedure. Now we turn our attention to understanding the conditions under which this general idea can be shown to converge.

A crucial question to ask is whether our algorithm maintains the contraction property that made value iteration so appealing in the first place - the property that allowed us to show convergence to a unique fixed point. We must be careful here because the contraction mapping theorem is specific to a given norm. In the case of value iteration, we showed the Bellman optimality operator is a contraction in the sup-norm, which aligns naturally with how we compare policies based on their value functions.

The situation becomes more complicated with fitted methods because we are not dealing with just a single operator. At each iteration, we perform exact, unbiased pointwise evaluations of the Bellman operator, but instead of obtaining the next function exactly, we get the closest representable one under our chosen function approximation scheme. A key insight from {cite:t}`Gordon1995` is that the fitting step can be conceptualized as an additional operator that gets applied on top of the exact Bellman operator to produce the next function parameters. This leads to viewing fitted value methods - which for simplicity we describe only for the value case, though the Q-value setting follows similarly - as the composition of two operators:

$$v_{n+1} = \Gamma(\mathrm{L}(v_n))$$

where $\mathrm{L}$ is the Bellman operator and $\Gamma$ represents the function approximation mapping.

Now we arrive at the central question: if $\mathrm{L}$ was a sup-norm contraction, is $\Gamma$ composed with $\mathrm{L}$ still a sup-norm contraction? What conditions must hold for this to be true? This question is fundamental because if we can establish that the composition of these two operators maintains the contraction property in the sup-norm, we get directly that our resulting successive approximation method will converge.

## The Search for Nonexpansive Operators

Consider what happens in the fitting step: we have two value functions $v$ and $w$, and after applying the Bellman operator $\mathrm{L}$ to each, we get new target values that differ by at most $\gamma$ times their original difference in sup-norm (due to $\mathrm{L}$ being a $\gamma$-contraction in the sup norm). But what happens when we fit to these target values? If the function approximator can exaggerate differences between its target values, even a small difference in the targets could lead to a larger difference in the fitted functions. This would be disastrous - even though the Bellman operator shrinks differences between value functions by a factor of $\gamma$, the fitting step could amplify them back up, potentially breaking the contraction property of the composite operator.

In order to ensure that the composite operator is contractive, we need conditions on $\Gamma$ such that if $\mathrm{L}$ is a sup-norm contraction then the composition also is. A natural property to consider is when $\Gamma$ is a non-expansion. By definition, this means that for any functions $v$ and $w$:

$$\|\Gamma(v) - \Gamma(w)\|_\infty \leq \|v - w\|_\infty$$

This turns out to be exactly what we need, since if $\Gamma$ is a non-expansion, then for any functions $v$ and $w$:

$$\|\Gamma(\mathrm{L}(v)) - \Gamma(\mathrm{L}(w))\|_\infty \leq \|L(v) - L(w)\|_\infty \leq \gamma\|v - w\|_\infty$$

The first inequality uses the non-expansion property of $\Gamma$, while the second uses the fact that $\mathrm{L}$ is a $\gamma$-contraction. Together they show that the composite operator $\Gamma \circ L$ remains a $\gamma$-contraction.

## Gordon's Averagers

But which function approximators satisfy this non-expansion property? Gordon shows that "averagers" - approximators that compute their outputs as weighted averages of their training values - are always non-expansions in sup-norm. This includes many common approximation schemes like k-nearest neighbors, linear interpolation, and kernel smoothing with normalized weights. The intuition is that if you're taking weighted averages with weights that sum to one, you can never extrapolate beyond the range of your training values -- these methods "interpolate".  This theoretical framework explains why simple interpolation methods like k-nearest neighbors have proven remarkably stable in practice, while more sophisticated approximators can fail catastrophically. It suggests a clear design principle: to guarantee convergence, we should either use averagers directly or modify other approximators to ensure they never extrapolate beyond their training targets.

More precisely, a function approximator $\Gamma$ is an averager if for any state $s$ and any target function $v$, the fitted value can be written as:

$$\Gamma(v)(s) = \sum_{i=1}^n w_i(s) v(s_i)$$

where the weights $w_i(s)$ satisfy:
1. $w_i(s) \geq 0$ for all $i$ and $s$
2. $\sum_{i=1}^n w_i(s) = 1$ for all $s$
3. The weights $w_i(s)$ depend only on $s$ and the training points $\{s_i\}$, not on the values $v(s_i)$

Let $m = \min_i v(s_i)$ and $M = \max_i v(s_i)$. Then:

$$m = m\sum_i w_i(s) \leq \sum_i w_i(s)v(s_i) \leq M\sum_i w_i(s) = M$$

So $\Gamma(v)(s) \in [m,M]$ for all $s$, meaning the fitted function cannot take values outside the range of its training values. This property is what makes averagers "interpolate" rather than "extrapolate" and is directly related to why they preserve the contraction property when composed with the Bellman operator. To see why averagers are non-expansions, consider two functions $v$ and $w$. At any state $s$:

$$\begin{align*}
|\Gamma(v)(s) - \Gamma(w)(s)| &= \left|\sum_{i=1}^n w_i(s)v(s_i) - \sum_{i=1}^n w_i(s)w(s_i)\right| \\
&= \left|\sum_{i=1}^n w_i(s)(v(s_i) - w(s_i))\right| \\
&\leq \sum_{i=1}^n w_i(s)|v(s_i) - w(s_i)| \\
&\leq \|v - w\|_\infty \sum_{i=1}^n w_i(s) \\
&= \|v - w\|_\infty
\end{align*}$$

Since this holds for all $s$, we have $\|\Gamma(v) - \Gamma(w)\|_\infty \leq \|v - w\|_\infty$, proving that $\Gamma$ is a non-expansion.

## Which Function Approximators Interpolate vs Extrapolate?

### K-nearest neighbors (KNN)

Let's look at specific examples, starting with k-nearest neighbors. For any state $s$, let $s_{(1)}, ..., s_{(k)}$ denote the k nearest training points to $s$. Then:

$$\Gamma(v)(s) = \frac{1}{k}\sum_{i=1}^k v(s_{(i)})$$

This is clearly an averager with weights $w_i(s) = \frac{1}{k}$ for the k nearest neighbors and 0 for all other points.

For kernel smoothing with a kernel function $K$, the fitted value is:

$$\Gamma(v)(s) = \frac{\sum_{i=1}^n K(s - s_i)v(s_i)}{\sum_{i=1}^n K(s - s_i)}$$

The denominator normalizes the weights to sum to 1, making this an averager with weights $w_i(s) = \frac{K(s - s_i)}{\sum_{j=1}^n K(s - s_j)}$.

### Linear Regression 

In contrast, methods like linear regression and neural networks can and often do extrapolate beyond their training targets. More precisely, given a dataset of state-value pairs $\{(s_i, v(s_i))\}_{i=1}^n$, these methods fit parameters to minimize some error criterion, and the resulting function $\Gamma(v)(s)$ may take values outside the interval $[\min_i v(s_i), \max_i v(s_i)]$ even when evaluated at a new state $s$. For instance, linear regression finds parameters by minimizing squared error:

$$\min_{\theta} \sum_{i=1}^n (v(s_i) - \theta^T\phi(s_i))^2$$

The resulting fitted function is:

$$\Gamma(v)(s) = \phi(s)^T(\Phi^T\Phi)^{-1}\Phi^T v$$

where $\Phi$ is the feature matrix with rows $\phi(s_i)^T$. This cannot be written as a weighted average with weights independent of $v$. Indeed, we can construct examples where the fitted value at a point lies outside the range of training values. For example, consider two sets of target values defined on just three points $s_1 = 0$, $s_2 = 1$, and $s_3 = 2$:

$$v = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}, \quad w = \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}$$

Using a single feature $\phi(s) = s$, our feature matrix is:

$$\Phi = \begin{bmatrix} 0 \\ 1 \\ 2 \end{bmatrix}$$

For function $v$, the fitted parameters are:

$$\theta_v = (\Phi^T\Phi)^{-1}\Phi^T v = \frac{1}{14}(2)$$

And for function $w$:

$$\theta_w = (\Phi^T\Phi)^{-1}\Phi^T w = \frac{1}{14}(8)$$

Now if we evaluate these fitted functions at $s = 3$ (outside our training points):

$$\Gamma(v)(3) = 3\theta_v = \frac{6}{14} \approx 0.43$$
$$\Gamma(w)(3) = 3\theta_w = \frac{24}{14} \approx 1.71$$

Therefore:

$$|\Gamma(v)(3) - \Gamma(w)(3)| = \frac{18}{14} > 1 = \|v - w\|_\infty$$

### Spline Interpolation

Linear interpolation between points -- the technique used earlier in this chapter -- is an averager since for any point $s$ between knots $s_i$ and $s_{i+1}$:

$$\Gamma(v)(s) = \left(\frac{s_{i+1}-s}{s_{i+1}-s_i}\right)v(s_i) + \left(\frac{s-s_i}{s_{i+1}-s_i}\right)v(s_{i+1})$$

The weights sum to 1 and are non-negative. However, cubic splines, despite their smoothness advantages, can violate the non-expansion property. To see this, consider fitting a natural cubic spline to three points:

$$s_1 = 0,\; s_2 = 1,\; s_3 = 2$$

with two different sets of values:

$$v = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \quad w = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$$

The natural cubic spline for $v$ will overshoot at $s \approx 0.5$ and undershoot at $s \approx 1.5$ due to its attempt to minimize curvature, giving values outside the range $[0,1]$. Meanwhile, $w$ fits a flat line at 0. Therefore:

$$\|v - w\|_\infty = 1$$

but 

$$\|\Gamma(v) - \Gamma(w)\|_\infty > 1$$

This illustrates a general principle: methods that try to create smooth functions by minimizing some global criterion (like curvature in splines) often sacrifice the non-expansion property to achieve their smoothness goals.

