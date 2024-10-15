
## Smooth Optimality Equations for Infinite-Horizon MDPs

While the standard Bellman optimality equations use the max operator to determine the best action, an alternative formulation known as the smooth or soft Bellman optimality equations replaces this with a softmax operator. This approach, originating from {cite}`rust1987optimal` and later rediscovered in the context of maximum entropy inverse reinforcement learning {cite}`ziebart2008maximum` and soft Q-learning {cite}`haarnoja2017reinforcement`, introduces a degree of stochasticity into the decision-making process.

In the infinite-horizon setting, the smooth Bellman optimality equations take the form:

$$ v(s) = \frac{1}{\beta} \log \sum_{a \in A_s} \exp\left(\beta\left(r(s, a) + \gamma \sum_{j \in S} p(j | s, a) v(j)\right)\right) $$

Adopting an operator-theoretic perspective, we can define a nonlinear operator $\mathrm{L}_\beta$ such that the smooth value function of an MDP is then the solution to the following fixed-point equation:

$$ (\mathrm{L}_\beta \mathbf{v})(s) = \frac{1}{\beta} \log \sum_{a \in \mathcal{A}_s} \exp\left(\beta\left(r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right)\right) $$

The smooth Bellman operator $\mathrm{L}_\beta$ maintains several key properties:

1. As $\beta \to \infty$, $\mathrm{L}_\beta$ converges to the standard Bellman operator $\mathrm{L}$.
2. $\mathrm{L}_\beta$ is a contraction mapping in the supremum norm, and therefore has a unique fixed point.
3. The fixed point of $\mathrm{L}_\beta$ is associated with the value function of a stochastic policy, where the probability of choosing action $a$ in state $s$ is given by the softmax distribution:

   $$ \pi(a|s) = \frac{\exp\left(\beta\left(r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right)\right)}{\sum_{a' \in \mathcal{A}_s} \exp\left(\beta\left(r(s,a') + \gamma \sum_{j \in \mathcal{S}} p(j|s,a') v(j)\right)\right)} $$

This is simply the generalization of what would otherwise be the argmax operator in the original optimality equation.

This formulation is interesting for several reasons:

1. Smoothness is a desirable property from an optimization standpoint. Unlike $\gamma$, we view $\beta$ as a hyperparameter of our algorithm, which we can control to achieve the desired level of accuracy.

2. While presented from an intuitive standpoint where we replace the max by the log-sum-exp (a smooth maximum) and the argmax by the softmax (a smooth argmax), this formulation can also be obtained from various other perspectives, offering theoretical tools and solution methods. For example, {cite}`rust1987optimal` derived this algorithm by considering a setting in which the rewards are stochastic and perturbed by a Gumbel noise variable. When considering the corresponding augmented state space and integrating the noise, we obtain smooth equations. This interpretation is leveraged by Rust for modeling purposes.

    There is also a way to obtain this equation by starting from the energy-based formulation often used in supervised learning, in which we convert an unnormalized probability distribution into a distribution using the softmax transformation. This is essentially what {cite}`ziebart2008maximum` did in their paper. Furthermore, this perspective bridges with the literature on probabilistic graphical models, in which we can now cast the problem of finding an optimal smooth policy into one of maximum likelihood estimation (an inference problem). This is the idea of control as inference, which also admits the converse - that of inference as control - used nowadays for deriving fast samples and amortized inference techniques using reinforcement learning {cite}`levine2018reinforcement`.

    Finally, it's worth noting that we can also derive this form by considering an entropy-regularized formulation in which we penalize for the entropy of our policy in the reward function term. This formulation admits a solution that coincides with the smooth Bellman equations {cite}`haarnoja2017reinforcement`.