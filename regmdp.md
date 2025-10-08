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

Dynamic programming methods suffer from the curse of dimensionality and can quickly become difficult to apply in practice. Not only this, we may also be dealing with large or continuous state or action spaces. We have seen so far that we could address this problem using discretization, or interpolation. These were already examples of approximate dynamic programming. In this chapter, we will see other forms of approximations meant to facilitate the optimization problem, either by approximating the optimality equations, the value function, or the policy itself.
Approximation theory is at the heart of learning methods, and fundamentally, this chapter will be about the application of learning ideas to solve complex decision-making problems.

# Smooth Bellman Optimality Equations

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

$$ d(a_t | s_t) = \frac{\exp(\beta (r(s_t, a_t) + \gamma \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) V_{t+1}(s_{t+1})))}{\sum_{a'_t} \exp(\beta (r(s_t, a'_t) + \gamma \sum_{s_{t+1}} p(s_{t+1} | s_t, a'_t) V_{t+1}(s_{t+1})))} $$ -->

## Regularized Markov Decision Processes

Regularized MDPs {cite}`geist2019` provide another perspective on how the smooth Bellman equations come to be. This framework offers a more general approach in which we seek to find optimal policies under the infinite horizon criterion while also accounting for a regularizer that influences the kind of policies we try to obtain.

Let's set up some necessary notation. First, recall that the policy evaluation operator for a stationary policy with decision rule $d$ is defined as:

$$ \mathrm{L}_d v = r_d + \gamma P_d v $$

where $r_d$ is the expected reward under policy $d$, $\gamma$ is the discount factor, and $P_d$ is the state transition probability matrix under $d$. A complementary object to the value function is the q-function (or Q-factor) representation:

$$ \begin{align*}
q_\gamma^{d^\infty}(s, a) &= r(s, a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v_\gamma^{d^\infty}(j) \\
v_\gamma^{d^\infty}(s) &= \sum_{a \in \mathcal{A}_s} d(a | s) q_\gamma^{d^\infty}(s, a) 
\end{align*} $$

The policy evaluation operator can then be written in terms of the q-function as:

$$ [\mathrm{L}_d v](s) = \langle d(\cdot | s), q(s, \cdot) \rangle $$

### Legendre-Fenchel Transform

The workhorse behind the theory of regularized MDPs is the Legendre-Fenchel transform, also known as the convex conjugate. For a strongly convex function $\Omega: \Delta_{\mathcal{A}} \rightarrow \mathbb{R}$, its Legendre-Fenchel transform $\Omega^*: \mathbb{R}^{\mathcal{A}} \rightarrow \mathbb{R}$ is defined as:

$$ \Omega^*(q(s, \cdot)) = \max_{d(\cdot|s) \in \Delta_{\mathcal{A}}} \langle d(\cdot | s), q(s, \cdot) \rangle - \Omega(d(\cdot | s)) $$

An important property of this transform is that it has a unique maximizing argument, given by the gradient of $\Omega^*$. This gradient is Lipschitz and satisfies:

$$ \nabla \Omega^*(q(s, \cdot)) = \arg\max_d \langle d(\cdot | s), q(s, \cdot) \rangle - \Omega(d(\cdot | s)) $$

An important example of a regularizer is the negative entropy, which gives rise to the smooth Bellman equations as we are about to see. 

## Regularized Bellman Operators

With these concepts in place, we can now define the regularized Bellman operators:

1. **Regularized Policy Evaluation Operator** $(\mathrm{L}_{d,\Omega})$:

   $$ [\mathrm{L}_{d,\Omega} v](s) = \langle q(s,\cdot), d(\cdot | s) \rangle - \Omega(d(\cdot | s)) $$

2. **Regularized Bellman Optimality Operator** $(\mathrm{L}_\Omega)$:
           
   $$ [\mathrm{L}_\Omega v](s) = [\max_d \mathrm{L}_{d,\Omega} v ](s) = \Omega^*(q(s, \cdot)) $$

It can be shown that the addition of a regularizer in these regularized operators still preserves the contraction properties, and therefore the existence of a solution to the optimality equations and the convergence of successive approximation.

The regularized value function of a stationary policy with decision rule $d$, denoted by $v_{d,\Omega}$, is the unique fixed point of the operator equation:

$$\text{find $v$ such that } \enspace v = \mathrm{L}_{d,\Omega} v$$

Under the usual assumptions on the discount factor and the boundedness of the reward, the value of a policy can also be found in closed form by solving for $v$ in the linear system of equations:

$$ (\mathbf{I} - \gamma P_d) v =  (r_d - \Omega(d)) $$

The associated state-action value function $q_{d,\Omega}$ is given by:

$$\begin{align*}
q_{d,\Omega}(s, a) &= r(s, a) + \sum_{j \in \mathcal{S}} \gamma p(j|s,a) v_{d,\Omega}(j) \\
v_{d,\Omega}(s) &= \sum_{a \in \mathcal{A}_s} d(a | s) q_{d,\Omega}(s, a) - \Omega(d(\cdot | s))
\end{align*} $$

The regularized optimal value function $v^*_\Omega$ is then the unique fixed point of $\mathrm{L}_\Omega$ in the fixed point equation:

$$\text{find $v$ such that } v = \mathrm{L}_\Omega v$$

The associated state-action value function $q^*_\Omega$ is given by:

$$ \begin{align*}
q^*_\Omega(s, a) &= r(s, a) + \sum_{j \in \mathcal{S}} \gamma p(j|s,a) v^*_\Omega(j) \\
v^*_\Omega(s) &= \Omega^*(q^*_\Omega(s, \cdot))\end{align*} $$

An important result in the theory of regularized MDPs is that there exists a unique optimal regularized policy. Specifically, if $d^*_\Omega$ is a conserving decision rule (i.e., $d^*_\Omega = \arg\max_d \mathrm{L}_{d,\Omega} v^*_\Omega$), then the randomized stationary policy $(d^*_\Omega)^\infty$ is the unique optimal regularized policy.

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

### Equivalence Between Smooth Bellman Equations and Entropy-Regularized MDPs

We have now seen two distinct ways to arrive at smooth Bellman equations. Earlier in this chapter, we introduced the logsumexp operator as a smooth approximation to the max operator, motivated by analytical tractability and the desire for differentiability. Just now, we derived the same equations through the lens of regularized MDPs, where we explicitly penalize the entropy of policies. Remarkably, these two perspectives are mathematically equivalent: solving the smooth Bellman equation with inverse temperature parameter $\beta$ yields exactly the same optimal value function and optimal policy as solving the entropy-regularized MDP with regularization strength $\alpha = 1/\beta$. The two formulations are not merely similar—they describe identical optimization problems.

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

The entropy bonus at each state, when averaged over the policy, becomes a per-action penalty proportional to the negative log probability of the action taken. This reformulation is more useful because the modified reward now depends only on the state, the action taken, and the probability assigned to that specific action by the policy—not on the entire distribution over actions.

This expression shows that entropy regularization is equivalent to adding a state-action dependent penalty term $-\alpha \ln \pi(a|s)$ to the reward. Intuititively, this terms amounts to paying a cost for low-entropy (deterministic) policies.

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

This is precisely the smooth Bellman equation we derived earlier using the logsumexp operator. The inverse temperature parameter $\beta$ controls how closely the logsumexp approximates the max: as $\beta \to \infty$, we recover the standard Bellman equation, while for finite $\beta$, we have a smooth approximation that corresponds to optimizing with entropy regularization strength $\alpha = 1/\beta$.

The optimal policy is:

$$
\pi^*(a|s) = \frac{\exp\left(\beta q^*(s,a)\right)}{\sum_{a'} \exp\left(\beta q^*(s,a')\right)} = \text{softmax}_\beta(q^*(s,\cdot))(a),
$$
which is exactly the softmax policy parametrized by inverse temperature.

The derivation establishes the complete equivalence: the value function $v^*$ that solves the smooth Bellman equation is identical to the optimal value function $v^*_\Omega$ of the entropy-regularized MDP (with $\Omega$ being negative entropy and $\alpha = 1/\beta$), and the softmax policy that is greedy with respect to this value function achieves the maximum of the entropy-regularized objective. Both approaches yield the same numerical solution—the same values at every state and the same policy prescriptions. The only difference is how we conceptualize the problem: as smoothing the Bellman operator for computational tractability, or as explicitly trading off reward maximization against policy entropy.

This equivalence has important implications. When we use smooth Bellman equations with a logsumexp operator, we are implicitly solving an entropy-regularized MDP. Conversely, when we explicitly add entropy regularization to an MDP objective, we arrive at smooth Bellman equations as the natural description of optimality. This dual perspective will prove valuable in understanding various algorithms and theoretical results. For instance, in soft actor-critic methods and other maximum entropy reinforcement learning algorithms, the connection between smooth operators and entropy regularization provides both computational benefits (differentiability) and conceptual clarity (why we want stochastic policies).