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
# Infinite-Horizon Markov Decision Processes

Finite-horizon stochastic dynamic programming obtains its boundary condition
from the terminal reward. What value equation remains when the process has no
terminal date? Discounting makes the infinite reward stream bounded and turns
the Bellman equations into fixed-point problems.

The undiscounted expected total reward of policy $\boldsymbol{\pi} \in
\Pi^{\mathrm{HR}}$ is

$$
v^{\boldsymbol{\pi}}(s) = \mathbb{E}\left[\sum_{t=1}^{\infty} r(S_t, A_t)\right]
$$

One drawback of this model is that we could easily encounter values that are $+\infty$ or $-\infty$, even in a setting as simple as a single-state MDP which loops back into itself and where the accrued reward is nonzero.

Therefore, it is often more convenient to work with an alternative formulation which guarantees the existence of a limit: the expected total discounted reward of policy $\boldsymbol{\pi} \in \Pi^{\mathrm{HR}}$ is defined to be:

$$
v_\gamma^{\boldsymbol{\pi}}(s) \equiv \lim_{N \rightarrow \infty} \mathbb{E}\left[\sum_{t=1}^N \gamma^{t-1} r(S_t, A_t)\right]
$$

for $0 \leq \gamma < 1$ and when $\max_{s \in \mathcal{S}} \max_{a \in \mathcal{A}_s}|r(s, a)| = R_{\max} < \infty$, in which case, $|v_\gamma^{\boldsymbol{\pi}}(s)| \leq (1-\gamma)^{-1} R_{\max}$.


Finally, another possibility for the infinite-horizon setting is the so-called average reward or gain of policy $\boldsymbol{\pi} \in \Pi^{\mathrm{HR}}$ defined as:

$$
g^{\boldsymbol{\pi}}(s) \equiv \lim_{N \rightarrow \infty} \frac{1}{N} \mathbb{E}\left[\sum_{t=1}^N r(S_t, A_t)\right]
$$

We won't be working with this formulation in this course due to its inherent practical and theoretical complexities. 

Extending the previous notion of optimality from finite-horizon models, a policy $\boldsymbol{\pi}^*$ is said to be discount optimal for a given $\gamma$ if: 

$$
v_\gamma^{\boldsymbol{\pi}^*}(s) \geq v_\gamma^{\boldsymbol{\pi}}(s) \quad \text { for each } s \in S \text { and all } \boldsymbol{\pi} \in \Pi^{\mathrm{HR}}
$$

Furthermore, the value of a discounted MDP, $v_\gamma^*(s)$, is defined by:

$$
v_\gamma^*(s) \equiv \max _{\boldsymbol{\pi} \in \Pi^{\mathrm{HR}}} v_\gamma^{\boldsymbol{\pi}}(s)
$$

More often, we refer to $v_\gamma$ by simply calling it the optimal value function. 

As for the finite-horizon setting, the infinite horizon discounted model does not require history-dependent policies, since for any $\boldsymbol{\pi} \in \Pi^{HR}$ there exists a $\boldsymbol{\pi}^{\prime} \in \Pi^{MR}$ with identical total discounted reward:
$$
v_\gamma^*(s) \equiv \max_{\boldsymbol{\pi} \in \Pi^{HR}} v_\gamma^{\boldsymbol{\pi}}(s)=\max_{\boldsymbol{\pi} \in \Pi^{MR}} v_\gamma^{\boldsymbol{\pi}}(s) .
$$

## Random Horizon Interpretation of Discounting

Can geometric termination give the discount factor a probabilistic meaning
rather than treating it only as an algebraic device?
The use of discounting can be motivated both from a modeling perspective and as a means to ensure that the total reward remains bounded. From the modeling perspective, we can view discounting as a way to put more or less weight on the immediate rewards vs. the long-term consequences. There is also another interpretation which stems from that of a finite horizon model but with an uncertain end time. More precisely:

Let $v_\nu^{\boldsymbol{\pi}}(s)$ denote the expected total reward obtained by using policy $\boldsymbol{\pi}$ when the horizon length $\nu$ is random. We define it by:

$$
v_\nu^{\boldsymbol{\pi}}(s) \equiv \mathbb{E}_s^{\boldsymbol{\pi}}\left[\mathbb{E}_\nu\left\{\sum_{t=1}^\nu r(S_t, A_t)\right\}\right]
$$


````{prf:theorem} Random horizon interpretation of discounting
:label: prop-5-3-1
Suppose that the horizon $\nu$ follows a geometric distribution with parameter $\gamma$, $0 \leq \gamma < 1$, independent of the policy such that 
$P(\nu=n) = (1-\gamma) \gamma^{n-1}, \, n=1,2, \ldots$, then $v_\nu^{\boldsymbol{\pi}}(s) = v_\gamma^{\boldsymbol{\pi}}(s)$ for all $s \in \mathcal{S}$ .
````

````{prf:proof}
See proposition 5.3.1 in {cite}`Puterman1994`.

By definition of the finite-horizon value function and the law of total expectation:

$$
v_\nu^{\boldsymbol{\pi}}(s) = \sum_{n=1}^{\infty} P(\nu=n) \cdot v_n^{\boldsymbol{\pi}}(s) = \sum_{n=1}^{\infty} (1-\gamma) \gamma^{n-1} \cdot E_s^{\boldsymbol{\pi}} \left\{\sum_{t=1}^n r(S_t, A_t)\right\}.
$$

Combining the expectation with the sum over $n$:

$$
v_\nu^{\boldsymbol{\pi}}(s) = E_s^{\boldsymbol{\pi}} \left\{\sum_{n=1}^{\infty} (1-\gamma) \gamma^{n-1} \sum_{t=1}^n r(S_t, A_t)\right\}.
$$

**Reordering the summations:** Under the bounded reward assumption $|r(s,a)| \leq R_{\max}$ and $\gamma < 1$, we have

$$
E_s^{\boldsymbol{\pi}} \left\{\sum_{n=1}^{\infty} \sum_{t=1}^n |r(S_t, A_t)| \cdot (1-\gamma) \gamma^{n-1}\right\} \leq R_{\max} \sum_{n=1}^{\infty} n (1-\gamma) \gamma^{n-1} = \frac{R_{\max}}{1-\gamma} < \infty,
$$
which justifies exchanging the order of summation by Fubini's theorem.

To reverse the order, note that the pair $(n,t)$ with $1 \leq t \leq n$ can be reindexed by fixing $t$ first and letting $n$ range from $t$ to $\infty$:

$$
\sum_{n=1}^{\infty} \sum_{t=1}^n = \sum_{t=1}^{\infty} \sum_{n=t}^{\infty}.
$$

Therefore:
\begin{align*}
v_\nu^{\boldsymbol{\pi}}(s) &= E_s^{\boldsymbol{\pi}} \left\{\sum_{t=1}^{\infty} r(S_t, A_t) \sum_{n=t}^{\infty} (1-\gamma) \gamma^{n-1}\right\}.
\end{align*}

**Evaluating the inner sum:** Using the substitution $m = n - t + 1$ (so $n = m + t - 1$):
\begin{align*}
\sum_{n=t}^{\infty} (1-\gamma) \gamma^{n-1} &= \sum_{m=1}^{\infty} (1-\gamma) \gamma^{m+t-2} \\
&= \gamma^{t-1} (1-\gamma) \sum_{m=1}^{\infty} \gamma^{m-1} \\
&= \gamma^{t-1} (1-\gamma) \cdot \frac{1}{1-\gamma} = \gamma^{t-1}.
\end{align*}

Substituting back:

$$
v_\nu^{\boldsymbol{\pi}}(s) = E_s^{\boldsymbol{\pi}} \left\{\sum_{t=1}^{\infty} \gamma^{t-1} r(S_t, A_t)\right\} = v_\gamma^{\boldsymbol{\pi}}(s).
$$

````


Under this geometric horizon the expected length is $\mathbb{E}[\nu] = 1/(1-\gamma)$, often called the **effective horizon** of the discounted problem: $\gamma = 0.99$ corresponds to planning about 100 steps ahead on average. The construction can be simulated directly. After collecting each reward, flip a coin that comes up heads with probability $\gamma$: continue on heads and end the episode on tails. The undiscounted return of such an episode is an unbiased estimate of $v_\gamma^{\boldsymbol{\pi}}(s)$. In practice we rarely simulate this way. Truncating episodes at random throws away the rewards that come after the coin lands on tails, which increases the variance of the estimate, and in learning it discards signal that could have been used. The equivalence is still useful to keep in mind, because it lets an algorithm trade a discount factor for a stopping probability and back.

### Drawing an MDP as a graph, and the cemetery state

Small MDPs are often drawn as directed graphs, as Puterman does for his toy examples. Each state is a node. Each action available in a state contributes arrows out of that node, one for every possible next state, labeled with the probability $p(j \mid s, a)$. The action sets may differ across states: in {numref}`fig-mdp-graph-absorbing`(a), state $s_1$ has two actions and state $s_2$ only one, as allowed by the state-dependent sets $\mathcal{A}_s$. Without the actions, such a picture is the transition graph of a Markov chain. Fixing a decision rule $\pi$ chooses (or randomizes over) the arrows out of each node and leaves exactly that Markov chain, whose transition matrix is the $\mathbf{P}_\pi$ of the next section. A controlled system is a family of Markov chains, one per decision rule, and $\mathbf{P}_\pi$ folds the choice of action into the transition probabilities.

:::{figure} _static/mdp-graph-absorbing.svg
:label: fig-mdp-graph-absorbing
:alt: Left, a two-state MDP drawn as a graph with action-labeled arrows. Right, the same graph with every probability multiplied by gamma and an extra absorbing state reached with probability one minus gamma from every state.

(a) A two-state MDP drawn as a graph. In $s_1$, action $a_1$ keeps the system in $s_1$ and action $a_2$ moves it to $s_2$; in $s_2$ only $a_1$ is available. (b) The same MDP with a cemetery state $\Delta$. Every original transition probability is multiplied by $\gamma$, and from every state the system moves to $\Delta$ with probability $1-\gamma$, whatever the action. $\Delta$ is absorbing and pays no reward.
:::

The graph gives a third reading of discounting. Add a new state $\Delta$, sometimes called a cemetery or sink state. From every state $s$ and every action $a$, move to $\Delta$ with probability $1-\gamma$, and otherwise follow the original dynamics, so that the new transition probabilities are $\tilde p(j \mid s,a) = \gamma\, p(j \mid s,a)$ for $j \in \mathcal{S}$ and $\tilde p(\Delta \mid s,a) = 1-\gamma$. Once in $\Delta$ the system stays there forever and collects reward zero. In this auxiliary MDP, the undiscounted expected total reward of any policy equals its discounted value in the original MDP, because the probability of still being alive at step $t$ is $\gamma^{t-1}$. This is the construction of {prf:ref}`prop-5-3-1` written as a state rather than as a random variable, and it is how discounted problems are reduced to total-reward problems with absorbing states, sometimes called stochastic shortest-path problems.

Not every process has an absorbing state. A system can run forever, and for its long-run behavior to be well defined we need some structure. The usual assumption is ergodicity: the chain $\mathbf{P}_\pi$ settles into a stationary distribution that spreads its mass over many states, and long-run averages are averages under that distribution. This is the setting of the average-reward criterion mentioned at the start of the chapter, which we do not pursue.

## Vector Representation in Markov Decision Processes

How do transition kernels, rewards, and policies become matrices and vectors
that expose the Bellman equations as operator equations?

Let V be the set of bounded real-valued functions on a discrete state space S. This means any function $ f \in V $ satisfies the condition:

$$
\|f\| = \max_{s \in S} |f(s)| < \infty.
$$
where the notation $ \|f\| $ represents the sup-norm (or $ \ell_\infty $-norm) of the function $ f $. 

When working with discrete state spaces, we can interpret elements of V as vectors and linear operators on V as matrices, allowing us to leverage tools from linear algebra. The sup-norm ($\ell_\infty$ norm) of matrix $\mathbf{H}$ is defined as:

$$
\|\mathbf{H}\| \equiv \max_{s \in S} \sum_{j \in S} |\mathbf{H}_{s,j}|
$$

where $\mathbf{H}_{s,j}$ represents the $(s, j)$-th component of the matrix $\mathbf{H}$.

For a Markovian decision rule $\pi \in \Pi^{MD}$, we define:

\begin{align*}
\mathbf{r}_\pi(s) &\equiv r(s, \pi(s)), \quad \mathbf{r}_\pi \in \mathbb{R}^{|S|}, \\
[\mathbf{P}_\pi]_{s,j} &\equiv p(j \mid s, \pi(s)), \quad \mathbf{P}_\pi \in \mathbb{R}^{|S| \times |S|}.
\end{align*}

For a randomized decision rule $\pi \in \Pi^{MR}$, these definitions extend to:

\begin{align*}
\mathbf{r}_\pi(s) &\equiv \sum_{a \in A_s} \pi(a \mid s) \, r(s, a), \\
[\mathbf{P}_\pi]_{s,j} &\equiv \sum_{a \in A_s} \pi(a \mid s) \, p(j \mid s, a).
\end{align*}

In both cases, $\mathbf{r}_\pi$ denotes a reward vector in $\mathbb{R}^{|S|}$, with each component $\mathbf{r}_\pi(s)$ representing the reward associated with state $s$. Similarly, $\mathbf{P}_\pi$ is a transition probability matrix in $\mathbb{R}^{|S| \times |S|}$, capturing the transition probabilities under decision rule $\pi$.

For a nonstationary Markovian policy $\boldsymbol{\pi} = (\pi_1, \pi_2, \ldots) \in \Pi^{MR}$, the expected total discounted reward is given by:

$$
\mathbf{v}_\gamma^{\boldsymbol{\pi}}(s)=\mathbb{E}\left[\sum_{t=1}^{\infty} \gamma^{t-1} r\left(S_t, A_t\right) \,\middle|\, S_1 = s\right].
$$

Using vector notation, and writing $\mathbf{P}_{\boldsymbol{\pi}}^{t-1} \equiv \mathbf{P}_{\pi_1} \mathbf{P}_{\pi_2} \cdots \mathbf{P}_{\pi_{t-1}}$ for the $(t-1)$-step transition matrix of the nonstationary policy (with $\mathbf{P}_{\boldsymbol{\pi}}^{0} \equiv \mathbf{I}$), this can be expressed as:

$$
\begin{aligned}
\mathbf{v}_\gamma^{\boldsymbol{\pi}} &= \sum_{t=1}^{\infty} \gamma^{t-1} \mathbf{P}_{\boldsymbol{\pi}}^{t-1} \mathbf{r}_{\pi_t} \\
&= \mathbf{r}_{\pi_1} + \gamma \mathbf{P}_{\pi_1} \mathbf{r}_{\pi_2} + \gamma^2 \mathbf{P}_{\pi_1} \mathbf{P}_{\pi_2} \mathbf{r}_{\pi_3} + \cdots \\
&= \mathbf{r}_{\pi_1} + \gamma \mathbf{P}_{\pi_1} \left( \mathbf{r}_{\pi_2} + \gamma \mathbf{P}_{\pi_2} \mathbf{r}_{\pi_3} + \gamma^2 \mathbf{P}_{\pi_2} \mathbf{P}_{\pi_3} \mathbf{r}_{\pi_4} + \cdots \right).
\end{aligned}
$$

Each term has a direct reading: $[\mathbf{P}_{\boldsymbol{\pi}}^{t-1}]_{s,j}$ is the probability of being in state $j$ at time $t$ when starting from $s$, so $\gamma^{t-1}\mathbf{P}_{\boldsymbol{\pi}}^{t-1}\mathbf{r}_{\pi_t}$ is the expected discounted reward collected $t-1$ steps ahead. The matrix products average over every path the system can take under the policy, which is what a Monte Carlo estimate would approximate by simulation.

Factoring $\gamma \mathbf{P}_{\pi_1}$ out of every term after the first gives a recursive relationship:

$$
\mathbf{v}_\gamma^{\boldsymbol{\pi}} = \mathbf{r}_{\pi_1} + \gamma \mathbf{P}_{\pi_1} \mathbf{v}_\gamma^{\boldsymbol{\pi}^{\prime}},
$$

where $\boldsymbol{\pi}^{\prime} = (\pi_2, \pi_3, \ldots)$ is the policy shifted by one step. Unrolling this recursion recovers the series above, so the series is the solution of the recursion.


For a stationary policy $\boldsymbol{\pi} = \mathrm{const}(\pi)$ with constant decision rule $\pi$, the total expected reward simplifies to:

$$
\begin{align*}
\mathbf{v}_\gamma^{\pi} &= \mathbf{r}_\pi+ \gamma \mathbf{P}_\pi \mathbf{v}_\gamma^{\pi} \\
&=\sum_{t=1}^{\infty} \gamma^{t-1} \mathbf{P}_\pi^{t-1} \mathbf{r}_{\pi}
\end{align*}
$$

This last expression is called a Neumann series expansion, and it's guaranteed to exist under the assumptions of bounded reward and discount factor strictly less than one. 

```{prf:theorem} Neumann Series and Invertibility
:label: neumann-series

The **spectral radius** of a matrix $\mathbf{H}$ is defined as:

$$
\rho(\mathbf{H}) \equiv \max_{i} |\lambda_i(\mathbf{H})|
$$

where $\lambda_i(\mathbf{H})$ are the eigenvalues of $\mathbf{H}$.

**Neumann Series Existence:** For any matrix $\mathbf{H}$, the Neumann series

$$
\sum_{t=0}^{\infty} \mathbf{H}^t = \mathbf{I} + \mathbf{H} + \mathbf{H}^2 + \cdots
$$

converges if and only if $\rho(\mathbf{H}) < 1$. When this condition holds, the matrix $(\mathbf{I} - \mathbf{H})$ is invertible and

$$
(\mathbf{I} - \mathbf{H})^{-1} = \sum_{t=0}^{\infty} \mathbf{H}^t.
$$

```
Note that for any induced matrix norm $\|\cdot\|$ (i.e., a norm satisfying $\|\mathbf{H}\mathbf{v}\| \leq \|\mathbf{H}\| \cdot \|\mathbf{v}\|$ for all vectors $\mathbf{v}$) and any matrix $\mathbf{H}$, the spectral radius is bounded by:

$$
\rho(\mathbf{H}) \leq \|\mathbf{H}\|.
$$


This inequality provides a practical way to verify the convergence condition $\rho(\mathbf{H}) < 1$ by checking the simpler condition $\|\mathbf{H}\| < 1$ rather than trying to compute the eigenvalues directly.

We can now verify that $(\mathbf{I} - \gamma \mathbf{P}_\pi)$ is invertible and the Neumann series converges.

1. **Norm of the transition matrix:** Since $\mathbf{P}_\pi$ is a stochastic matrix (each row sums to 1 and all entries are non-negative), its $\ell_\infty$-norm is:

   $$
   \|\mathbf{P}_\pi\| = \max_{s \in S} \sum_{j \in S} [\mathbf{P}_\pi]_{s,j} = \max_{s \in S} 1 = 1.
   $$

2. **Norm of the scaled matrix:** Using the homogeneity property of norms, we have:

   $$
   \|\gamma \mathbf{P}_\pi\| = |\gamma| \cdot \|\mathbf{P}_\pi\| = |\gamma| \cdot 1 = |\gamma|.
   $$

3. **Bounding the spectral radius:** Since the spectral radius is bounded by the matrix norm:

   $$
   \rho(\gamma \mathbf{P}_\pi) \leq \|\gamma \mathbf{P}_\pi\| = |\gamma|.
   $$

4. **Verifying convergence:** Since $0 \leq \gamma < 1$ by assumption, we have:

   $$
   \rho(\gamma \mathbf{P}_\pi) \leq |\gamma| < 1.
   $$
   
   This strict inequality guarantees that $(\mathbf{I} - \gamma \mathbf{P}_\pi)$ is invertible and the Neumann series converges.

Therefore, the Neumann series expansion converges and yields:

$$
\mathbf{v}_\gamma^{\pi} = (\mathbf{I} - \gamma \mathbf{P}_\pi)^{-1} \mathbf{r}_\pi = \sum_{t=0}^{\infty} (\gamma \mathbf{P}_\pi)^t \mathbf{r}_\pi = \sum_{t=1}^{\infty} \gamma^{t-1} \mathbf{P}_\pi^{t-1} \mathbf{r}_\pi.
$$

The middle sum is indexed from $t=0$ to match the statement of {prf:ref}`neumann-series`, while the last one keeps the convention $S_1 = s$ used for the value function. They are the same series, shifted by one index.

Consequently, for a stationary policy, $\mathbf{v}_\gamma^{\pi}$ can be determined as the solution to the linear equation:

$$
\mathbf{v} = \mathbf{r}_\pi+ \gamma \mathbf{P}_\pi\mathbf{v},
$$

which can be rearranged to:

$$
(\mathbf{I} - \gamma \mathbf{P}_\pi) \mathbf{v} = \mathbf{r}_\pi.
$$

We can also characterize $\mathbf{v}_\gamma^{\pi}$ as the solution to an operator equation. More specifically, define the transformation $\mathrm{L}_\pi$ by

$$
\mathrm{L}_\pi \mathbf{v} \equiv \mathbf{r}_\pi+\gamma \mathbf{P}_\pi\mathbf{v}
$$

for any $\mathbf{v} \in V$. Intuitively, $\mathrm{L}_\pi$ takes a value function $\mathbf{v}$ as input and returns a new value function that combines immediate rewards ($\mathbf{r}_\pi$) with discounted future values ($\gamma \mathbf{P}_\pi\mathbf{v}$). 

```{note}
While we often refer to $\mathrm{L}_\pi$ as a "linear operator" in the RL literature, it is technically an **affine operator** (or affine transformation), not a linear operator in the strict sense. To see why, recall that a linear operator $\mathcal{T}$ must satisfy:

1. **Additivity:** $\mathcal{T}(\mathbf{v}_1 + \mathbf{v}_2) = \mathcal{T}(\mathbf{v}_1) + \mathcal{T}(\mathbf{v}_2)$
2. **Homogeneity:** $\mathcal{T}(\alpha \mathbf{v}) = \alpha \mathcal{T}(\mathbf{v})$ for all scalars $\alpha$

However, $\mathrm{L}_\pi$ fails the additivity test:

$$
\mathrm{L}_\pi(\mathbf{v}_1 + \mathbf{v}_2) = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi(\mathbf{v}_1 + \mathbf{v}_2) = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi\mathbf{v}_1 + \gamma \mathbf{P}_\pi\mathbf{v}_2
$$

while

$$
\mathrm{L}_\pi(\mathbf{v}_1) + \mathrm{L}_\pi(\mathbf{v}_2) = (\mathbf{r}_\pi + \gamma \mathbf{P}_\pi\mathbf{v}_1) + (\mathbf{r}_\pi + \gamma \mathbf{P}_\pi\mathbf{v}_2) = 2\mathbf{r}_\pi + \gamma \mathbf{P}_\pi\mathbf{v}_1 + \gamma \mathbf{P}_\pi\mathbf{v}_2.
$$

The presence of the constant term $\mathbf{r}_\pi$ makes $\mathrm{L}_\pi$ affine rather than linear. An affine operator has the form $\mathcal{A}(\mathbf{v}) = \mathbf{b} + \mathcal{T}(\mathbf{v})$, where $\mathbf{b}$ is a constant vector and $\mathcal{T}$ is a linear operator. In our case, $\mathbf{b} = \mathbf{r}_\pi$ and $\mathcal{T}(\mathbf{v}) = \gamma \mathbf{P}_\pi\mathbf{v}$.

Despite this technical distinction, the term "linear operator" is commonly used in the reinforcement learning literature when referring to $\mathrm{L}_\pi$, following a slight abuse of terminology.
```

Therefore, we view $\mathrm{L}_\pi$ as an operator mapping elements of $V$ to $V$: i.e., $\mathrm{L}_\pi: V \rightarrow V$. The fact that the value function of a policy is the solution to a fixed-point equation can then be expressed with the statement: 

$$
\mathbf{v}_\gamma^{\pi}=\mathrm{L}_\pi \mathbf{v}_\gamma^{\pi}.
$$

This is a **fixed-point equation**: the value function $\mathbf{v}_\gamma^{\pi}$ is a fixed point of the operator $\mathrm{L}_\pi$.

## Solving Operator Equations

Which iterative and Newton-like methods solve a fixed point when direct matrix
inversion is unavailable or inappropriate?

The operator equation we encountered in MDPs, $\mathbf{v}_\gamma^{\pi} = \mathrm{L}_\pi \mathbf{v}_\gamma^{\pi}$, is a specific instance of a more general class of problems known as operator equations. These equations appear in various fields of mathematics and applied sciences, ranging from differential equations to functional analysis.

Operator equations can take several forms, each with its own characteristics and solution methods:

1. **Fixed Point Form**: $x = \mathrm{T}(x)$, where $\mathrm{T}: X \rightarrow X$.
   This form is common in fixed-point problems, such as our MDP equation: we seek a fixed point $x^*$ such that $x^* = \mathrm{T}(x^*)$.

2. **General Operator Equation**: $\mathrm{T}(x) = y$, where $\mathrm{T}: X \rightarrow Y$.
   Here, $X$ and $Y$ can be different spaces. We seek an $x \in X$ that satisfies the equation for a given $y \in Y$.

3. **Nonlinear Equation**: $\mathrm{T}(x) = 0$, where $\mathrm{T}: X \rightarrow Y$.
   A special case of the general operator equation where we seek roots or zeros of the operator.

4. **Variational Inequality**: Find $x^* \in K$ such that $\langle \mathrm{T}(x^*), x - x^* \rangle \geq 0$ for all $x \in K$.
   Here, $K$ is a closed convex subset of $X$, and $\mathrm{T}: K \rightarrow X^*$ (the dual space of $X$). These problems often arise in optimization, game theory, and partial differential equations.

:::{iframe} ../interactive/fixed-point-lab.html
:label: fig-fixed-point-lab
:width: 100%
:title: Fixed-point lab
:class: fixed-point-lab-replay

The same iteration on four unknowns: a number, a pendulum trajectory, the value
of a one-state MDP and the water value function of the reservoir. Each method
draws a line through $(x, \mathrm{T}(x))$ with some slope $s$ and jumps to where
that line meets $y = x$. Slope zero gives successive approximation and value
iteration. The slope of $\mathrm{T}$ gives Newton's method and policy iteration.
:::

### Successive Approximation Method

For equations in fixed-point form, the basic tool is successive approximation, also called fixed-point iteration: start from any guess and apply the operator repeatedly. Whether this works depends on the operator, and the standard sufficient condition is that it be a contraction.

````{prf:definition} Contraction mapping
:label: def-contraction-mapping

Let $(X, \|\cdot\|)$ be a normed space. An operator $\mathrm{T}: X \rightarrow X$ is a **contraction** with respect to $\|\cdot\|$ if there exists a constant $q \in [0,1)$, the **contraction factor**, such that for all $x, y \in X$:

$$
\|\mathrm{T}(x) - \mathrm{T}(y)\| \leq q \, \|x - y\|.
$$
````

In other words, a contraction is a Lipschitz map whose Lipschitz constant is strictly less than one. The property is attached to a norm: an operator can be a contraction in one norm and fail to be one in another. For MDPs the norm that works is the sup-norm $\|\cdot\|_\infty$, and from here on $\|\cdot\|$ without a subscript means the sup-norm. When we later say that $\mathrm{L}_\pi$ or $\Bellman$ is a contraction, the precise statement is that it is a **sup-norm contraction**. Results proved in the sup-norm do not carry over to other norms for free; that a Bellman operator composed with a projection need not be a contraction in a weighted $\ell_2$ norm is one of the technical roots of the instability that Sutton calls the deadly triad.

````{prf:theorem} Banach fixed-point theorem
:label: banach-fixed-point

Let $(X, \|\cdot\|)$ be a Banach space (a complete normed space) and let $\mathrm{T}: X \rightarrow X$ be a contraction with factor $q \in [0,1)$. Then:

1. $\mathrm{T}$ has a **unique** fixed point $x^* = \mathrm{T}(x^*)$.
2. For any $x_0 \in X$, the sequence $x_{n+1} = \mathrm{T}(x_n)$ converges to $x^*$.
3. The error satisfies the a priori bound $\|x_n - x^*\| \leq \frac{q^n}{1-q}\|x_1 - x_0\|$ and the a posteriori bound $\|x_{n+1} - x^*\| \leq \frac{q}{1-q}\|x_{n+1} - x_n\|$.
````

The theorem is an existence and uniqueness statement, and its proof is constructive: the sequence used to show that a fixed point exists is the algorithm that computes it. Uniqueness matters, because nothing about an equation $x = \mathrm{T}(x)$ alone rules out many solutions. The theorem gives a sufficient condition only. An operator that is not a contraction may still have a fixed point, and other fixed-point theorems cover some of those cases.

:::{iframe} ../interactive/contraction-cloud.html
:label: fig-contraction-cloud
:width: 100%
:title: Contraction cloud
:class: contraction-cloud-replay

What a contraction shrinks. Each dot is a value function $v = (v(1), v(2))$ of a
two-state MDP, and each step applies the Bellman optimality operator (defined
later in this chapter) to all of them at once. The dashed box, whose side is the
cloud's diameter in the sup-norm, shrinks by the factor $\gamma$ at every step,
and the whole cloud collapses onto the unique fixed point $v^*$.
:::

````{prf:algorithm} Successive Approximation
:label: successive-approximation

**Input:** An operator $\mathrm{T}: X \rightarrow X$, an initial guess $x_0 \in X$, and a tolerance $\epsilon > 0$  
**Output:** An iterate $x_{n+1}$ whose last step satisfies $\|x_{n+1} - x_n\| < \epsilon$

1. Initialize $n = 0$  
2. **repeat**  
    3. Compute $x_{n+1} = \mathrm{T}(x_n)$  
    4. If $\|x_{n+1} - x_n\| < \epsilon$, **return** $x_{n+1}$  
    5. Set $n = n + 1$  
6. **until** convergence or maximum iterations reached  

````

The stopping test only measures the size of the last step, a quantity computed from the iterates themselves. On its own it says nothing about the distance to the fixed point. Turning it into such a guarantee takes extra work, and this is what the a posteriori bound in {prf:ref}`banach-fixed-point` provides for a contraction: stopping when $\|x_{n+1} - x_n\| < \epsilon$ guarantees $\|x_{n+1} - x^*\| < \frac{q}{1-q}\epsilon$. When $q$ is close to one, a small step can still leave the iterate far from $x^*$. For MDPs we will choose the tolerance with this in mind, so that value iteration returns an $\varepsilon$-optimal policy ({prf:ref}`value-iteration-convergence`). Since the error shrinks by at least a factor $q$ per step, the convergence rate is linear.

However, the contraction mapping condition is not the only one that can lead to convergence. For instance, if $\mathrm{T}$ is nonexpansive (i.e., Lipschitz continuous with Lipschitz constant 1) and $X$ is a Banach space with certain geometrical properties (e.g., uniformly convex), then under additional conditions (e.g., $\mathrm{T}$ has at least one fixed point), the successive approximation method can still converge, albeit potentially more slowly than in the contraction case.

In practice, when dealing with specific problems like MDPs or differential equations, the properties of the operator often naturally align with one of these convergence conditions. For example, in discounted MDPs, the Bellman operator is a contraction in the supremum norm, which guarantees the convergence of value iteration.

For policy evaluation, successive approximation replaces the linear solve with the loop $\mathbf{v}_{k+1} = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}_k$. A direct solve of $(\mathbf{I} - \gamma \mathbf{P}_\pi)\mathbf{v} = \mathbf{r}_\pi$ costs on the order of $|S|^3$ operations, while each iteration costs one matrix-vector product, and the loop never needs to form or factor a matrix. Here $\mathrm{L}_\pi$ is a contraction with factor $\gamma$, by the same norm argument we used for the Neumann series.

The second general tool for operator equations is the Newton-Kantorovich method, which extends Newton's method to function spaces. We postpone it to {ref}`sec-newton-kantorovich-pi`: applied to the Bellman optimality equation it yields policy iteration, just as successive approximation applied to the same equation yields value iteration.

## Optimality Equations for Infinite-Horizon MDPs

How does actionwise maximization turn fixed-policy evaluation into the Bellman
optimality equation?

Recall that in the finite-horizon setting, the optimality equations are:

$$
v_n(s) = \max_{a \in A_s} \left\{r(s, a) + \gamma \sum_{j \in S} p(j | s, a) v_{n+1}(j)\right\}
$$

where $v_n(s)$ is the value function at time step $n$ for state $s$, $A_s$ is the set of actions available in state $s$, $r(s, a)$ is the reward function, $\gamma$ is the discount factor, and $p(j | s, a)$ is the transition probability from state $s$ to state $j$ given action $a$.

Intuitively, we would expect that by taking the limit of $n$ to infinity, we might get the nonlinear equations:

$$
v(s) = \max_{a \in A_s} \left\{r(s, a) + \gamma \sum_{j \in S} p(j | s, a) v(j)\right\}
$$

which are called the optimality equations or Bellman equations for infinite-horizon MDPs.

We can adopt an operator-theoretic perspective by defining operators on the space $V$ of bounded real-valued functions on the state space $S$. For a deterministic Markov rule $\pi \in \Pi^{MD}$, define the **policy-evaluation operator**:

$$
(\BellmanPi v)(s) = r(s,\pi(s)) + \gamma \sum_{j \in \mathcal{S}} p(j|s,\pi(s)) v(j)
$$

Terminology varies here. Bertsekas and Sutton and Barto call $\BellmanPi$ the Bellman operator for $\pi$, and part of the RL theory literature calls it the Bellman consistency operator. In this book we keep the name Bellman operator for the operator with the maximum, defined next, which is closer to Bellman's own usage, and call $\BellmanPi$ the policy-evaluation operator because evaluating a policy is what it is used for.

The **Bellman optimality operator** is then:

$$
\Bellman \mathbf{v} \equiv \max_{\pi \in \Pi^{MD}} \left\{\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}\right\}
$$

where $\Pi^{MD}$ is the set of Markov deterministic decision rules, $\mathbf{r}_\pi$ is the reward vector under decision rule $\pi$, and $\mathbf{P}_\pi$ is the transition probability matrix under decision rule $\pi$.

Note that while we write $\max_{\pi \in \Pi^{MD}}$, we do not implement the above operator by enumerating all decision rules. Rather, the fact that we compare policies based on their value functions in a componentwise fashion means that maximizing over the space of Markovian deterministic rules reduces to the following update in component form:

$$
(\Bellman \mathbf{v})(s) = \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right\}
$$

The two forms of $\Bellman$, a maximum over decision rules and a maximum over actions in each state, are equal. The following proposition makes this precise.

```{prf:proposition}
The operator $\Bellman$ defined as a maximization over Markov deterministic decision rules:

$$(\Bellman \mathbf{v})(s) = \max_{\pi \in \Pi^{MD}} \left\{r(s,\pi(s)) + \gamma \sum_{j \in \mathcal{S}} p(j|s,\pi(s)) v(j)\right\}$$

is equivalent to the componentwise maximization over actions:

$$(\Bellman \mathbf{v})(s) = \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right\}$$
```

```{prf:proof}
Fix $s$. Let 

$$
Q_v(s,a) \triangleq r(s,a)+\gamma\sum_{j}p(j\mid s,a)\,v(j).
$$

For any rule $\pi \in \Pi^{MD}$, we have $(\BellmanPi v)(s)=Q_v(s,\pi(s))\le \max_{a\in\mathcal{A}_s}Q_v(s,a)$.

Taking the maximum over $\pi$ gives

$$
\max_{\pi\in\Pi^{MD}}(\BellmanPi v)(s) \le \max_{a\in\mathcal{A}_s}Q_v(s,a).
$$

Conversely, choose a **greedy selector** $\pi^v\in\Pi^{MD}$ such that for each $s$,

$$\pi^v(s)\in\arg\max_{a\in\mathcal{A}_s}Q_v(s,a)$$

(possible since $\mathcal{A}_s$ is finite; otherwise use a measurable $\varepsilon$-greedy selector). Then

$$
(\Bellman _{\pi^v}v)(s)=Q_v(s,\pi^v(s))=\max_{a\in\mathcal{A}_s}Q_v(s,a),
$$

so $\max_{\pi}(\BellmanPi v)(s)\ge \max_{a}Q_v(s,a)$. Combining both inequalities yields equality.
```

For convenience, we define the **greedy selector** $\mathrm{Greedy}(v) \in \Pi^{MD}$ that extracts a decision rule from a value function:

$$
\mathrm{Greedy}(v)(s) \in \arg\max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right\}
$$

Equivalently, $\pi = \mathrm{Greedy}(v)$ is any decision rule with $\BellmanPi v = \Bellman v$. Puterman calls such a rule **$v$-improving**. A $v^*_\gamma$-improving rule, one that is greedy with respect to the optimal value function, is called **conserving**. In the RL literature both are simply called **greedy** policies: given $v$, the policy can act myopically because $v$ already summarizes the future. This selector will be useful for expressing algorithms succinctly:
- **Value iteration:** iterate $v_{k+1} = \Bellman v_k$, then extract $\pi = \mathrm{Greedy}(v_k)$ at termination
- **Policy iteration:** $\pi_{k+1} = \mathrm{Greedy}(v^{\pi_k})$ with $v^{\pi_k}$ solving $v = \mathrm{L}_{\pi_k}v$

### Optimality of Deterministic Stationary Policies

The definition of optimality at the start of the chapter compares $\boldsymbol{\pi}^*$ with every policy in $\Pi^{\mathrm{HR}}$: history-dependent, nonstationary and randomized. The comparison is componentwise, so $\boldsymbol{\pi}^*$ must be at least as good as every other policy in every starting state. Searching that class directly is hopeless. A history-dependent rule $\pi_t(a \mid h_t)$ with $h_t = (s_1, a_1, \ldots, a_{t-1}, s_t)$ depends on a history that grows with $t$, and a nonstationary policy needs one rule per stage, which for an infinite horizon means infinitely many. Even restricted to deterministic Markov policies, enumerating every policy and evaluating each one is out of reach. The next result shows that nothing is lost by searching a much smaller set.

````{prf:theorem} Optimality equation and stationary optimal policies
:label: thm-stationary-optimal

Suppose $0 \leq \gamma < 1$, rewards are bounded, and $S$ and every $\mathcal{A}_s$ are finite. Then:

1. $\Bellman$ is a sup-norm contraction with factor $\gamma$, so by {prf:ref}`banach-fixed-point` the optimality equation $v = \Bellman v$ has a unique solution, and this solution is the optimal value function $v^*_\gamma$.
2. Any conserving decision rule $\pi^* \in \mathrm{Greedy}(v^*_\gamma)$ defines a deterministic stationary policy $\mathrm{const}(\pi^*)$ that is optimal over all of $\Pi^{\mathrm{HR}}$:

$$
v_\gamma^{\pi^*} = v^*_\gamma = \sup_{\boldsymbol{\pi} \in \Pi^{\mathrm{HR}}} v_\gamma^{\boldsymbol{\pi}}.
$$
````

See {cite}`Puterman1994`, Section 6.2, for the proof. The contraction property is shown in the proof of {prf:ref}`value-iteration-convergence` below. Two consequences are worth stating. First, the supremum over $\Pi^{\mathrm{HR}}$ is attained, by a deterministic stationary policy, so we may restrict the search to the set $\Pi^{MD}$ of deterministic Markov decision rules applied at every stage. Second, the optimal value function is unique but the optimal policy need not be: when several actions attain the maximum in some state, each choice gives an optimal policy. Note also that the theorem is specific to the discounted infinite-horizon case. In a finite-horizon problem the optimal policy depends on the stage in general.

### Q-Functions

The optimality equations can also be written in terms of a function of a state and an action. The **optimal Q-function** (called *Q-factors* in the operations research literature) is the unique solution of

$$
q(s,a) = r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j \mid s,a) \max_{a' \in \mathcal{A}_j} q(j,a').
$$

As with $v$, the right-hand side defines an operator on functions of $(s,a)$, and later chapters write it with the same symbol, $q^* = \Bellman q^*$. It is a sup-norm contraction with factor $\gamma$ when $\gamma < 1$, so $q^*$ exists and is unique. The two optimal functions determine each other:

$$
v^*_\gamma(s) = \max_{a \in \mathcal{A}_s} q^*(s,a), \qquad q^*(s,a) = r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j \mid s,a)\, v^*_\gamma(j),
$$

and a conserving rule can be read off directly as $\pi^*(s) \in \arg\max_{a \in \mathcal{A}_s} q^*(s,a)$. Probabilistically, $q^*$ conditions on both the state and the first action, while $v^*_\gamma$ averages the action out. Optimality is still defined through $v$, because that is how policies are ordered.

When $\mathbf{P}$ and $r$ are known, as in the examples of this chapter, there is no reason to prefer $q$: computing $v$ is cheaper and the greedy rule follows from it. The Q-function matters in model-free RL, where we observe sampled transitions and rewards but not the transition probabilities. Extracting a greedy action from $v$ requires the expectation $\sum_j p(j \mid s,a)\, v(j)$, which needs the model. Extracting it from $q$ only requires a maximum over the action argument, which works for a learned $q$, a neural network for instance, without knowing $p$. This is why the Q-function form is the starting point of the [simulation-based chapters](monte-carlo-bellman-estimation.md). The convenience has a price. A Q-function has one value per state-action pair instead of one per state, and it keeps the detail that $v$ averages away, so learning $q^*$ typically takes more capacity, more samples and better coverage of the actions than learning $v^*_\gamma$ would.

## Algorithms for Solving the Optimality Equations

What computational trade-off separates value iteration, policy evaluation,
and policy improvement?

The optimality equations are operator equations. Therefore, we can apply general numerical methods to solve them. Applying the successive approximation method to the Bellman optimality equation yields a method known as "value iteration" in dynamic programming. Following Puterman and the older operations research literature, we reserve this name for the iteration with the max. The same iteration applied to the affine operator $\mathrm{L}_\pi$ is iterative policy evaluation, although RL texts often call any iteration of this form value iteration. A direct application of the blueprint for successive approximation yields the following algorithm:

````{prf:algorithm} Value Iteration
:label: value-iteration

**Input** Given an MDP $(S, A, P, R, \gamma)$ and tolerance $\varepsilon > 0$  

**Output** Compute an $\varepsilon$-optimal value function $v$ and policy $\pi$  

1. Initialize $v_0(s) = 0$ for all $s \in S$  
2. $n \leftarrow 0$  
3. **repeat**  

    1. For each $s \in S$:  

        1. $v_{n+1}(s) \leftarrow (\Bellman v_n)(s) = \max_{a \in A} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)v_n(j)\right\}$  

    2. $\delta \leftarrow \|v_{n+1} - v_n\|_\infty$  
    3. $n \leftarrow n + 1$  

4. **until** $\delta < \frac{\varepsilon(1-\gamma)}{2\gamma}$  
5. Extract greedy policy: $\pi \leftarrow \mathrm{Greedy}(v_n)$ where

    $$\mathrm{Greedy}(v)(s) \in \arg\max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)v(j)\right\}$$

6. **return** $v_n, \pi$  
````

The termination criterion in this algorithm is based on a specific bound that provides guarantees on the quality of the solution. This is in contrast to supervised learning, where we often use arbitrary termination criteria based on computational budget or early stopping when the learning curve flattens. This is because establishing implementable generalization bounds in supervised learning is challenging.

However, in the dynamic programming context, we can derive various bounds that can be implemented in practice. These bounds help us terminate our procedure with a guarantee on the precision of our value function and, correspondingly, on the optimality of the resulting policy.

This detail is easy to miss. The value iteration in {cite:t}`SuttonBarto2018` stops when the change falls below a small threshold $\theta$, without relating $\theta$ to the quality of the returned policy. The threshold $\varepsilon(1-\gamma)/(2\gamma)$ above is chosen so that the answer comes with a certificate: you fix the accuracy $\varepsilon$ in advance, the loop runs until the test passes, and the greedy policy it returns is then guaranteed to be $\varepsilon$-optimal ({prf:ref}`value-iteration-convergence`). Two caveats apply. The guarantee concerns the exact arithmetic; floating-point rounding is not part of the analysis, and with $\gamma$ close to one a very small $\varepsilon$ may be below what float32 or even float64 can resolve. And whether a certificate matters depends on the application. Nobody needs one to solve a grid world, but an economist who reports a policy's monetary value, or an engineer who must document a safety margin, does.

The value function is only an instrument. What we want in the end is a policy, and given $v^*_\gamma$ (or any approximation of it) we obtain one by the greedy step $\mathrm{Greedy}(v)$: for each state, compute the one-step lookahead $r(s,a) + \gamma \sum_j p(j \mid s,a) v(j)$ for every action and keep the best. With arrays of shape $(|\mathcal{S}|, |\mathcal{A}|)$ this is a single `argmax` along the action axis. Nothing is simulated forward, because $v$ already summarizes the future.

````{prf:proposition} Convergence of Value Iteration 
:label: value-iteration-convergence
(Adapted from {cite:t}`Puterman1994` theorem 6.3.1)

Let $v_0$ be any initial value function, $\varepsilon > 0$ a desired accuracy, and let $\{v_n\}$ be the sequence of value functions generated by value iteration, i.e., $v_{n+1} = \Bellman v_n$ for $n \geq 0$, where $\Bellman$ is the Bellman optimality operator. Then:

1. $v_n$ converges to the optimal value function $v^*_\gamma$,
2. The algorithm terminates in finite time,
3. The resulting policy $\pi_\varepsilon$ is $\varepsilon$-optimal, and
4. When the algorithm terminates, $v_{n+1}$ is within $\varepsilon/2$ of $v^*_\gamma$.

````

````{prf:proof}
Parts 1 and 2 follow directly from the fact that $\Bellman$ is a contraction mapping. Hence, by Banach's fixed-point theorem, it has a unique fixed point (which is $v^*_\gamma$), and repeated application of $\Bellman$ will converge to this fixed point. Moreover, this convergence happens at a geometric rate, which ensures that we reach the termination condition in finite time.

To show that the Bellman optimality operator $\Bellman$ is a contraction mapping, we need to prove that for any two value functions $v$ and $u$:

$$\|\Bellman v - \Bellman u\|_\infty \leq \gamma \|v - u\|_\infty$$

where $\gamma \in [0,1)$ is the discount factor and $\|\cdot\|_\infty$ is the supremum norm.

Let's start by writing out the definition of $\Bellman v$ and $\Bellman u$:

$$\begin{align*}
(\Bellman v)(s) &= \max_{a \in A} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)v(j)\right\}\\
(\Bellman u)(s) &= \max_{a \in A} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)u(j)\right\}
\end{align*}$$

Fix a state $s$. Let $a_v$ be an action that achieves the maximum in $(\Bellman v)(s)$, and $a_u$ an action that achieves the maximum in $(\Bellman u)(s)$. We bound the difference $(\Bellman v)(s) - (\Bellman u)(s)$ from above and from below, one direction at a time.

**Upper bound.** Since $a_v$ is the maximizer for $v$, $(\Bellman v)(s)$ is *equal* to the value of $a_v$:

$$(\Bellman v)(s) = r(s,a_v) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a_v)v(j).$$

The action $a_v$ need not be the maximizer for $u$, so plugging it into the maximum that defines $(\Bellman u)(s)$ can only give a smaller or equal value:

$$(\Bellman u)(s) \geq r(s,a_v) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a_v)u(j).$$

Subtracting the second line from the first, the rewards $r(s,a_v)$ cancel because both lines use the same action, and the result is an upper bound because we subtract a lower bound on $(\Bellman u)(s)$:

$$(\Bellman v)(s) - (\Bellman u)(s) \leq \gamma \sum_{j \in \mathcal{S}} p(j|s,a_v)\bigl(v(j) - u(j)\bigr).$$

**Lower bound.** Swap the roles of $v$ and $u$ and use $a_u$ instead. Now $(\Bellman u)(s)$ is equal to the value of $a_u$ under $u$, while $(\Bellman v)(s)$ is at least the value of $a_u$ under $v$:

$$\begin{align*}
(\Bellman u)(s) &= r(s,a_u) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a_u)u(j),\\
(\Bellman v)(s) &\geq r(s,a_u) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a_u)v(j).
\end{align*}$$

Subtracting in the same way:

$$(\Bellman u)(s) - (\Bellman v)(s) \leq \gamma \sum_{j \in \mathcal{S}} p(j|s,a_u)\bigl(u(j) - v(j)\bigr).$$

**Bounding each sum by the max-norm.** Each sum is an average of the differences $v(j) - u(j)$ (or $u(j) - v(j)$) with weights $p(j|s,a) \geq 0$ that sum to $1$. An average is never larger than its largest term, and every term satisfies $v(j) - u(j) \leq |v(j) - u(j)| \leq \|v - u\|_\infty$. Hence, for either action $a \in \{a_v, a_u\}$:

$$\sum_{j \in \mathcal{S}} p(j|s,a)\bigl(v(j) - u(j)\bigr) \leq \sum_{j \in \mathcal{S}} p(j|s,a)\,\|v - u\|_\infty = \|v - u\|_\infty,$$

and the same holds with $u(j) - v(j)$ in place of $v(j) - u(j)$. Plugging this into the two bounds above gives

$$(\Bellman v)(s) - (\Bellman u)(s) \leq \gamma \|v - u\|_\infty \quad\text{and}\quad (\Bellman u)(s) - (\Bellman v)(s) \leq \gamma \|v - u\|_\infty.$$

A number $x$ with $x \leq c$ and $-x \leq c$ satisfies $|x| \leq c$, so

$$|(\Bellman v)(s) - (\Bellman u)(s)| \leq \gamma \|v - u\|_\infty.$$

Since this holds for all $s \in \mathcal{S}$, taking the supremum over $s$ gives:

$$\|\Bellman v - \Bellman u\|_\infty \leq \gamma \|v - u\|_\infty$$

Thus, $\Bellman$ is a contraction mapping with contraction factor $\gamma$.

Now, let's prove parts 3 and 4. Suppose the algorithm has just terminated, i.e., $\|v_{n+1} - v_n\|_\infty < \frac{\varepsilon(1-\gamma)}{2\gamma}$ for some $n$. We want to show that our current value function $v_{n+1}$ and the policy $\pi_\varepsilon$ derived from it are close to optimal.

By the triangle inequality:

$$\|v^{\pi_\varepsilon}_\gamma - v^*_\gamma\|_\infty \leq \|v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty + \|v_{n+1} - v^*_\gamma\|_\infty$$

For the first term, since $v^{\pi_\varepsilon}_\gamma$ is the fixed point of $\mathrm{L}_{\pi_\varepsilon}$ and $\pi_\varepsilon$ is greedy with respect to $v_{n+1}$ (i.e., $\mathrm{L}_{\pi_\varepsilon}v_{n+1} = \Bellman v_{n+1}$):

$$
\begin{aligned}
\|v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty &= \|\mathrm{L}_{\pi_\varepsilon}v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty \\
&\leq \|\mathrm{L}_{\pi_\varepsilon}v^{\pi_\varepsilon}_\gamma - \mathrm{L}_{\pi_\varepsilon}v_{n+1}\|_\infty + \|\mathrm{L}_{\pi_\varepsilon}v_{n+1} - v_{n+1}\|_\infty \\
&= \|\mathrm{L}_{\pi_\varepsilon}v^{\pi_\varepsilon}_\gamma - \mathrm{L}_{\pi_\varepsilon}v_{n+1}\|_\infty + \|\Bellman v_{n+1} - v_{n+1}\|_\infty \\
&\leq \gamma\|v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty + \gamma\|v_{n+1} - v_n\|_\infty
\end{aligned}
$$

where we used that both $\Bellman$ and $\mathrm{L}_{\pi_\varepsilon}$ are contractions with factor $\gamma$, and that $v_{n+1} = \Bellman v_n$.

Rearranging:

$$\|v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty \leq \frac{\gamma}{1-\gamma}\|v_{n+1} - v_n\|_\infty$$

Similarly, since $v^*_\gamma$ is the fixed point of $\Bellman$:

$$\|v_{n+1} - v^*_\gamma\|_\infty = \|\Bellman v_n - \Bellman v^*_\gamma\|_\infty \leq \gamma\|v_n - v^*_\gamma\|_\infty \leq \frac{\gamma}{1-\gamma}\|v_{n+1} - v_n\|_\infty$$

Since $\|v_{n+1} - v_n\|_\infty < \frac{\varepsilon(1-\gamma)}{2\gamma}$:

$$\|v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty \leq \frac{\gamma}{1-\gamma} \cdot \frac{\varepsilon(1-\gamma)}{2\gamma} = \frac{\varepsilon}{2}$$

$$\|v_{n+1} - v^*_\gamma\|_\infty \leq \frac{\gamma}{1-\gamma} \cdot \frac{\varepsilon(1-\gamma)}{2\gamma} = \frac{\varepsilon}{2}$$

Combining these:

$$\|v^{\pi_\varepsilon}_\gamma - v^*_\gamma\|_\infty \leq \frac{\varepsilon}{2} + \frac{\varepsilon}{2} = \varepsilon$$

This completes the proof, showing that $v_{n+1}$ is within $\varepsilon/2$ of $v^*_\gamma$ (part 4) and $\pi_\varepsilon$ is $\varepsilon$-optimal (part 3).
````

### A Sharper Stopping Rule: the Span Seminorm

A good stopping rule stops as early as possible while still delivering the requested accuracy. The sup-norm test is not the best one available. A sharper test measures the change between iterates with the **span seminorm**

$$
\operatorname{sp}(u) = \max_{s} u(s) - \min_{s} u(s).
$$

It is a seminorm rather than a norm because $\operatorname{sp}(c\mathbf{1}) = 0$ for every constant $c$: the span ignores a uniform shift of all values. Let $d = v_{n+1} - v_n$ be the last change made by value iteration, with smallest and largest components $m = \min_s d(s)$ and $M = \max_s d(s)$. Two properties of $\Bellman$ give two-sided bounds. It is monotone ($u \le w$ componentwise implies $\Bellman u \le \Bellman w$), and it shifts constants by $\gamma$: $\Bellman(u + c\mathbf{1}) = \Bellman u + \gamma c \mathbf{1}$. From $v_{n+1} \ge v_n + m\mathbf{1}$ we get $\Bellman v_{n+1} \ge v_{n+1} + \gamma m \mathbf{1}$, and repeating the argument,

$$
v_{n+1} + \frac{\gamma}{1-\gamma}\, m\, \mathbf{1} \;\le\; v^*_\gamma \;\le\; v_{n+1} + \frac{\gamma}{1-\gamma}\, M\, \mathbf{1}.
$$

The same argument with $\mathrm{L}_\pi$ in place of $\Bellman$, for $\pi = \mathrm{Greedy}(v_{n+1})$, shows that $v^{\pi}_\gamma \ge v_{n+1} + \frac{\gamma}{1-\gamma} m \mathbf{1}$. Subtracting,

$$
\|v^*_\gamma - v^{\pi}_\gamma\|_\infty \le \frac{\gamma}{1-\gamma}\operatorname{sp}(v_{n+1} - v_n).
$$

So stopping when $\operatorname{sp}(v_{n+1} - v_n) < \varepsilon(1-\gamma)/\gamma$ also returns an $\varepsilon$-optimal greedy policy ({cite:t}`Puterman1994`, Section 6.6). Since $\operatorname{sp}(d) \le 2\|d\|_\infty$, this test is never later than the sup-norm test, and it is often much earlier: value iteration tends to spend many iterations shifting all values by nearly the same amount, which the span ignores. On a random MDP with 30 states, 4 actions, $\gamma = 0.95$ and rewards around 5, the sup-norm test with $\varepsilon = 10^{-3}$ needs 242 iterations and the span test needs 11, and both return an optimal policy. The bounds also give a better value estimate than $v_{n+1}$ itself: the midpoint $v_{n+1} + \frac{\gamma}{1-\gamma}\frac{m+M}{2}\mathbf{1}$ is within $\frac{\gamma}{2(1-\gamma)}\operatorname{sp}(d)$ of $v^*_\gamma$.

### Bellman contraction laboratory

Use the controls below to change the discount factor, transition persistence, reward asymmetry, and starting value. Before moving $\gamma$, predict how it will change the slope of the error envelope. The middle panel compares the observed error and Bellman residual with the contraction bound; the text below reports the final greedy policy.

```{marimo-config}
:echo: false
:error: false
:pyproject:

requires-python = ">=3.12"
dependencies = ["matplotlib", "numpy"]
```

```{marimo} python
import marimo as mo
import matplotlib.pyplot as plt
import numpy as np

discount = mo.ui.slider(start=0.10, stop=0.99, step=0.01, value=0.90, label="Discount γ")
persistence = mo.ui.slider(start=0.50, stop=0.99, step=0.01, value=0.85, label="Transition persistence")
reward_asymmetry = mo.ui.slider(start=-2.0, stop=2.0, step=0.1, value=0.8, label="Reward asymmetry")
initial_value = mo.ui.slider(start=-10.0, stop=10.0, step=0.5, value=0.0, label="Initial value scale")
mo.vstack([discount, persistence, reward_asymmetry, initial_value])
```

```{marimo} python
gamma_lab = discount.value
p_lab = persistence.value
reward_gap_lab = reward_asymmetry.value

transitions_lab = np.array([
    [[p_lab, 1 - p_lab], [1 - p_lab, p_lab]],
    [[1 - p_lab, p_lab], [p_lab, 1 - p_lab]],
])
rewards_lab = np.array([
    [1.0 + reward_gap_lab, 0.0],
    [0.0, 1.0 - reward_gap_lab],
])

def bellman_lab(value_lab):
    q_lab = rewards_lab + gamma_lab * np.einsum("asj,j->sa", transitions_lab, value_lab)
    return q_lab.max(axis=1), q_lab

v_star_lab = np.zeros(2)
for _ in range(1000):
    next_star_lab, _ = bellman_lab(v_star_lab)
    if np.max(np.abs(next_star_lab - v_star_lab)) < 1e-12:
        break
    v_star_lab = next_star_lab

value_lab = np.array([initial_value.value, -initial_value.value], dtype=float)
trace_lab = [value_lab.copy()]
residual_lab = []
error_lab = [np.max(np.abs(value_lab - v_star_lab))]
for _ in range(30):
    next_value_lab, _ = bellman_lab(value_lab)
    residual_lab.append(np.max(np.abs(next_value_lab - value_lab)))
    value_lab = next_value_lab
    trace_lab.append(value_lab.copy())
    error_lab.append(np.max(np.abs(value_lab - v_star_lab)))

trace_lab = np.asarray(trace_lab)
error_lab = np.asarray(error_lab)
residual_lab = np.asarray(residual_lab)
bound_lab = error_lab[0] * gamma_lab ** np.arange(error_lab.size)
_, final_q_lab = bellman_lab(value_lab)
policy_lab = final_q_lab.argmax(axis=1)
```

```{marimo} python
fig_lab, axes_lab = plt.subplots(1, 2, figsize=(10, 3.6))
axes_lab[0].plot(trace_lab[:, 0], label="state 0")
axes_lab[0].plot(trace_lab[:, 1], label="state 1")
axes_lab[0].axhline(v_star_lab[0], color="C0", linestyle=":", alpha=0.7)
axes_lab[0].axhline(v_star_lab[1], color="C1", linestyle=":", alpha=0.7)
axes_lab[0].set(xlabel="Iteration", ylabel="Value", title="Value-iteration trace")
axes_lab[0].legend()

axes_lab[1].semilogy(error_lab, label="actual error")
axes_lab[1].semilogy(bound_lab, linestyle="--", label="contraction bound")
axes_lab[1].semilogy(np.arange(1, residual_lab.size + 1), residual_lab, linestyle=":", label="Bellman residual")
axes_lab[1].set(xlabel="Iteration", ylabel="Sup norm", title="Error certificate")
axes_lab[1].legend()
fig_lab.tight_layout()

mo.vstack([
    fig_lab,
    mo.md(
        f"**Greedy policy:** state 0 → action {policy_lab[0]}, "
        f"state 1 → action {policy_lab[1]}.  "
        f"Final residual: **{residual_lab[-1]:.2e}**; "
        f"final error: **{error_lab[-1]:.2e}**."
    ),
])
```

:::{figure} _static/bellman-contraction-fallback.png
:label: fig-bellman-contraction-fallback
:class: pdf-fallback
:alt: Static Bellman contraction laboratory preview

Static preview of value-iteration traces and a geometric contraction bound. The online book provides controls for $\gamma$, transitions, rewards, and the initial value.
:::

(sec-policy-iteration)=
### Policy Iteration

Value iteration searches the space of value functions and reads a policy off at the end, so it reaches the policy indirectly. Policy iteration searches the space of policies directly. Bellman called this idea approximation in policy space, and {cite:t}`Howard1960` turned it into the algorithm below. Each iteration has two steps. The first evaluates the current policy, which is the linear problem we already know how to solve, so any policy-evaluation routine can be reused as a subroutine. The second improves the policy by acting greedily with respect to the value just computed.

````{prf:algorithm} Policy Iteration
:label: policy-iteration-standard

**Input:** MDP $(S, A, P, R, \gamma)$
**Output:** Optimal policy $\pi^*$

1. Initialize: $n = 0$, select an arbitrary decision rule $\pi_0 \in \Pi^{MD}$
2. **repeat**
   3. (Policy evaluation) Obtain $\mathbf{v}^n$ by solving:
   
      $$(\mathbf{I}-\gamma \mathbf{P}_{\pi_n}) \mathbf{v} = \mathbf{r}_{\pi_n}$$

   4. (Policy improvement) Choose $\pi_{n+1} = \mathrm{Greedy}(\mathbf{v}^n)$ where:

       $$\pi_{n+1} \in \arg\max_{\pi \in \Pi^{MD}}\left\{\mathbf{r}_\pi+\gamma \mathbf{P}_\pi \mathbf{v}^n\right\}$$
       
       equivalently, for each $s$:
       
       $$\pi_{n+1}(s) \in \arg\max_{a \in \mathcal{A}_s}\left\{r(s,a)+\gamma \sum_j p(j|s,a) \mathbf{v}^n(j)\right\}$$
       
       Set $\pi_{n+1} = \pi_n$ if possible.

   5. If $\pi_{n+1} = \pi_n$, **return** $\pi^* = \pi_n$

   6. $n = n + 1$
7. **until** convergence
````

Recall that $\mathbf{P}_{\pi_n}$ and $\mathbf{r}_{\pi_n}$ are built by averaging over the actions the rule selects: $[\mathbf{P}_{\pi}]_{s,j} = \sum_a \pi(a \mid s)\, p(j \mid s,a)$ and $\mathbf{r}_\pi(s) = \sum_a \pi(a \mid s)\, r(s,a)$. The tie-breaking instruction in step 4 ("keep $\pi_n$ if it is among the maximizers") matters for the stopping test. Without it, the algorithm could cycle among equally good greedy rules and never see the same rule twice.

````{prf:theorem} Finite termination of policy iteration
:label: thm-pi-finite

Suppose $S$ and every $\mathcal{A}_s$ are finite and $0 \le \gamma < 1$. Then the values produced by policy iteration are nondecreasing, $\mathbf{v}^{n+1} \ge \mathbf{v}^n$, and the algorithm stops after finitely many iterations with an optimal policy.
````

````{prf:proof}
Since $\pi_{n+1}$ is greedy with respect to $\mathbf{v}^n$, $\mathrm{L}_{\pi_{n+1}} \mathbf{v}^n = \Bellman \mathbf{v}^n \ge \mathrm{L}_{\pi_n}\mathbf{v}^n = \mathbf{v}^n$. Rearranged, $\mathbf{r}_{\pi_{n+1}} \ge (\mathbf{I} - \gamma \mathbf{P}_{\pi_{n+1}})\mathbf{v}^n$. The inverse $(\mathbf{I} - \gamma \mathbf{P}_{\pi_{n+1}})^{-1} = \sum_t (\gamma\mathbf{P}_{\pi_{n+1}})^t$ has nonnegative entries, so multiplying both sides by it preserves the inequality and gives $\mathbf{v}^{n+1} \ge \mathbf{v}^n$. The improvement is strict in some state unless $\Bellman \mathbf{v}^n = \mathbf{v}^n$, in which case $\mathbf{v}^n = v^*_\gamma$ and $\pi_n$ is optimal. A strictly improving sequence never repeats a rule, and there are finitely many deterministic Markov rules, so the algorithm must stop. See {cite:t}`Puterman1994`, Section 6.4.
````

Two remarks follow from the proof. First, the stopping rule is exact. When the same rule appears twice, $\mathbf{v}^n$ solves the optimality equation and the policy is optimal, not merely $\varepsilon$-optimal. This argument counts policies, so it is specific to finite state and action spaces; for continuous spaces, convergence needs the Newton-Kantorovich machinery of {ref}`sec-newton-kantorovich-pi`. Second, the proof never used that $\pi_{n+1}$ is greedy in *every* state. Any $\pi_{n+1}$ with $\mathrm{L}_{\pi_{n+1}}\mathbf{v}^n \ge \mathbf{v}^n$, strictly in at least one state, improves the policy. Improving a single state per iteration is enough for convergence. This matters later, because many RL algorithms contain a policy-iteration loop whose improvement step is only partial: a small change at each step, perhaps only in the states that were visited.

In practice policy iteration converges in very few iterations, often a handful, even for large problems. The reason is that it is Newton's method in disguise, a connection made by {cite:t}`PutermanBrumelle1979` and developed in {ref}`sec-newton-kantorovich-pi`. Each iteration costs more than a value-iteration step, though. Both algorithms pay about $|\mathcal{S}|^2|\mathcal{A}|$ operations for a greedy step, but policy iteration adds a direct policy evaluation, a linear solve of order $|\mathcal{S}|^3$.

### Modified Policy Iteration

Policy evaluation can also be done by successive approximation, and nothing forces us to run that inner loop to convergence. Stopping it after a few sweeps gives **modified policy iteration** (MPI) {cite}`PutermanShin1978`. One way to see what this does is through the Neumann series. Write the exact evaluation of the greedy rule $\pi = \mathrm{Greedy}(\mathbf{v})$ as a correction to the current guess:

$$
\mathbf{v}^{\pi}_\gamma = \mathbf{v} + (\mathbf{I} - \gamma\mathbf{P}_\pi)^{-1}(\Bellman \mathbf{v} - \mathbf{v}) = \mathbf{v} + \sum_{k=0}^{\infty} (\gamma \mathbf{P}_\pi)^k\, (\Bellman \mathbf{v} - \mathbf{v}).
$$

The step is the Bellman residual $\Bellman\mathbf{v} - \mathbf{v}$, propagated through the inverse. MPI truncates the series after $m+1$ terms:

$$
\mathbf{v}^{n+1} = \mathbf{v}^n + \sum_{k=0}^{m} (\gamma \mathbf{P}_{\pi_{n+1}})^k\, (\Bellman \mathbf{v}^n - \mathbf{v}^n) = (\mathrm{L}_{\pi_{n+1}})^{m+1}\, \mathbf{v}^n,
$$

where the second form, $m+1$ sweeps of policy evaluation started from $\mathbf{v}^n$, is how it is implemented. The integer $m \ge 0$ is a hyperparameter that interpolates between the two algorithms we have seen. With $m = 0$ the update is $\mathbf{v}^{n+1} = \mathrm{L}_{\pi_{n+1}}\mathbf{v}^n = \Bellman\mathbf{v}^n$, which is value iteration. With $m = \infty$ it is exact evaluation, which is policy iteration. In between, each iteration costs $m+1$ matrix-vector products and no linear solve.

````{prf:algorithm} Modified Policy Iteration
:label: alg-modified-policy-iteration

**Input:** MDP $(S, A, P, R, \gamma)$, number of evaluation sweeps $m \ge 0$, tolerance $\varepsilon > 0$  
**Output:** An $\varepsilon$-optimal policy

1. Initialize $\mathbf{v}^0$ with $\Bellman\mathbf{v}^0 \ge \mathbf{v}^0$, for example $\mathbf{v}^0 = \frac{\min_{s,a} r(s,a)}{1-\gamma}\mathbf{1}$; set $n = 0$
2. **repeat**
    1. (Improvement) $\pi_{n+1} \leftarrow \mathrm{Greedy}(\mathbf{v}^n)$, so that $\mathrm{L}_{\pi_{n+1}}\mathbf{v}^n = \Bellman\mathbf{v}^n$
    2. If $\|\Bellman\mathbf{v}^n - \mathbf{v}^n\|_\infty < \frac{\varepsilon(1-\gamma)}{2\gamma}$, **return** $\mathrm{Greedy}(\Bellman\mathbf{v}^n)$
    3. (Partial evaluation) $\mathbf{u} \leftarrow \Bellman\mathbf{v}^n$; repeat $m$ times: $\mathbf{u} \leftarrow \mathbf{r}_{\pi_{n+1}} + \gamma\mathbf{P}_{\pi_{n+1}}\mathbf{u}$
    4. $\mathbf{v}^{n+1} \leftarrow \mathbf{u}$; $n \leftarrow n+1$
````

The stopping test is the one from value iteration, and it gives the same guarantee, provided the returned policy is greedy with respect to $\Bellman\mathbf{v}^n$ (one more greedy step), exactly as value iteration returns the policy greedy with respect to its last iterate. Started from a $\mathbf{v}^0$ with $\Bellman\mathbf{v}^0 \ge \mathbf{v}^0$, the iterates increase monotonically to $v^*_\gamma$ and the returned policy is $\varepsilon$-optimal ({cite:t}`Puterman1994`, Section 6.5). The number of sweeps $m$ can also change from one iteration to the next.

Why should a truncated evaluation work at all? In Newton's method, the inverse Hessian (or inverse Jacobian) rescales the gradient step, a step-size schedule adapted to the local curvature. Here $(\mathbf{I} - \gamma\mathbf{P}_\pi)^{-1}$ plays that role for the Bellman residual, and value iteration replaces it by the identity. Truncating the Neumann series replaces it by an approximation, in the same way that quasi-Newton and inexact Newton methods use approximate Hessians or approximate linear solves. {ref}`sec-newton-kantorovich-pi` makes this precise.

## Exact Scheduling MDP for Inference Serving

How can a large request-level system be reduced to a finite MDP, and which
predictive distinctions disappear in that reduction?

The inference examples have so far treated the scheduling rule as fixed and the
GPU clock as the action. In a different decision interface, the clock is fixed
and the action is the phase that receives the next unit of service. Prefill
processes a waiting prompt and moves its request into the decode batch. Decode
advances every request in that batch by one step of token generation. Serving
either phase delays the other.

The trade-off comes from batching. A decode step costs about the same time and
power whether it advances one request or six, because generating a token is
limited by reading the model weights from memory, not by arithmetic. On the
measured NVIDIA L4 run used below, one decode stream produces a token every
57.0 ms; with four concurrent streams each stream still produces a token every
59.5 ms, and with eight every 60.8 ms. Six streams therefore finish requests
almost six times as fast as one. A scheduler that decodes as soon as one
request is ready keeps the batch small and wastes that capacity. A scheduler
that keeps prefilling to grow the batch leaves the requests already in it
waiting. The MDP decides where to switch.

An exact request-level Markov state would contain every prompt length, generated
token count, cache allocation, and waiting time. For computation, these
quantities are aggregated into the finite state

$$
s=(p,d)\in\{0,\ldots,6\}^2,
$$

where $p$ counts waiting prefill jobs and $d$ counts active decode jobs. The
actions are

$$
\mathcal A=\{\text{prefill},\text{decode},\text{idle}\}.
$$

An action is masked when its phase is empty. Prefill is also masked at $d=6$,
which represents the cache limit in this abstraction. The whole policy is
therefore one $7\times 7$ table, which the figure below draws in full.

Time advances in decision periods of 0.1 seconds. A Bernoulli arrival with
probability $\lambda$ per period is calibrated from the load-normalized version
of the same five-minute Azure trace used in the modeling chapter. The phase
rates and powers come from one measured NVIDIA L4 run of
Qwen/Qwen2.5-7B-Instruct served by vLLM 0.28.0, at the 1,125 MHz requested
clock level; its batch-balanced median realized graphics clock was 939.375 MHz.

- **Prefill.** The measured prefill rate and the trace's mean prompt length
  give the probability $q_{\mathrm p}$ that one prefill action completes a
  prompt. A completed prompt moves from $p$ to $d$.
- **Decode.** The single-stream time per output token and the trace's mean
  output length give the probability $q_1$ that a lone decode stream finishes
  its response in one period. With $d$ active streams, each stream finishes
  independently with probability $1-(1-q_1)^{\eta_d}$, where
  $\eta_d=\mathrm{TPOT}(1)/\mathrm{TPOT}(d)$ is the measured slowdown of one
  stream in a batch of $d$, interpolated between the measured batch sizes 1, 4,
  and 8. The number of completions is binomial, so its mean grows almost
  linearly in $d$.
- **Arrivals and drops.** After service, an arrival joins the prefill queue. An
  arrival that finds $p=6$ is dropped.

These choices define a complete transition matrix $P_{ss'}^u$ on 49 states.

The one-step cost counts the requests in the system, dropped arrivals, and
energy:

$$
c(s,u)=p+d
+100\,\mathbb E[N_{\mathrm{drop}}\mid s,u]
+0.1\frac{E(u)}{E_{\max}}.
$$

By Little's law, the long-run average of $p+d$ divided by the arrival rate is
the mean time a request spends in the system, so the first term is a latency
cost. A dropped request costs as much as holding one request for 100 periods,
or 10 seconds. That is about twice the mean time a request spends in the system
under the optimal policy, so dropping a request is never a cheap way to shorten
the queue. For prefill, decode, and idle, respectively, the measured profile
gives $E(u)=(6.427,6.274,2.339)$ joules per decision period. Each value is the
measured phase-power summary at the requested 1,125 MHz level multiplied by
0.1 seconds, rather than a direct request-level energy measurement. Only the
ratio $E(u)/E_{\max}$ enters the stage cost.

With $\gamma=0.99$, cost-minimizing value iteration applies

$$
\begin{aligned}
Q_n(s,u)&=c(s,u)+\gamma\sum_{s'}P_{ss'}^uV_n(s'),\\
V_{n+1}(s)&=\min_{u\in\mathcal A(s)}Q_n(s,u).
\end{aligned}
$$

Iteration stops when $\lVert V_{n+1}-V_n\rVert_\infty<10^{-10}$. The final
Bellman residual is checked independently and must be below $10^{-8}$.

The figure compares the value-iteration policy with two transparent rules.
**Prefill priority** prefills whenever a prompt waits and the batch has room.
**Decode priority** decodes whenever a decode job is active. Each replay
applies one policy for 60 seconds from an empty system. All three replays use
the same random numbers for arrivals and service, so they see the same
requests, and every difference between them comes from the decisions.

```{code-cell} python
:tags: [remove-input]
:label: fig-inference-scheduling-dp
:caption: Exact value iteration on the measured-L4-calibrated inference-scheduling MDP, compared with prefill priority and decode priority. Each policy is drawn over all 49 states. The panel beside it reports the policy's exact discounted cost from the empty system and its smallest gap $V^\pi(s)-V^*(s)$ over all states. The replay samples queue transitions from the reduced model at its fixed measured-profile clock; these trajectories are simulated, not direct vLLM observations.

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from inference_replay import render_serving_replay

display(HTML(render_serving_replay(
    Path("artifacts/inference_serving/textbook_results.json"),
    view="scheduling",
)))
```

:::{figure} _static/inference_serving/scheduling.svg
:label: fig-inference-scheduling-dp-fallback
:class: pdf-fallback
:alt: Policy tables for exact value iteration, prefill priority, and decode priority in the 49-state scheduling MDP.

The exact policy, prefill priority, and decode priority over all 49 states,
with each policy's exact discounted cost from the empty system. The
interactive version adds controls for playing, stepping through, and scrubbing
an episode sampled from the reduced transition model under each policy. The
queue trajectory is simulated rather than observed directly from vLLM.
:::

The optimal policy is a switching curve. When one or two prompts wait, it
prefills until four requests are decoding and then decodes. When three or more
wait, it fills the batch to five before decoding. It decodes whenever no prompt
waits, and idles only in the empty state. Decode priority is the opposite
extreme. It prefills only when the batch is empty, so it never runs more than
one stream, and a lone stream finishes fewer requests per period (0.059) than
arrive (0.071). Its prompt queue
therefore fills, and arrivals are dropped.

How can a reader check that value iteration returned the optimal policy rather
than a plausible one? For a fixed policy $\pi$, the policy evaluation equation
$V^\pi=c_\pi+\gamma P_\pi V^\pi$ is a linear system with 49 unknowns, and
solving it gives the exact discounted cost of $\pi$ in every state. An optimal
policy satisfies $V^*(s)\le V^\pi(s)$ in every state for every stationary
policy $\pi$, so each heuristic must lose everywhere, not only on average.

```{code-cell} python
:tags: [remove-input]

import json

import pandas as pd

scheduling = json.loads(
    Path("artifacts/inference_serving/textbook_results.json").read_text()
)["scheduling"]
names = {
    "exact_dp": "exact DP",
    "prefill_priority": "prefill priority",
    "decode_priority": "decode priority",
}
evaluations = {
    "exact_dp": scheduling["exact_policy_evaluation"],
    **scheduling["heuristic_evaluation"],
}
pd.DataFrame(
    {
        names[name]: {
            "cost from empty, exact": f"{check['cost_from_empty']:.2f}",
            "smallest gap V^π − V* over 49 states": f"{max(check['minimum_gap'], 0.0):.2f}",
            "largest gap V^π − V* over 49 states": f"{max(check['maximum_gap'], 0.0):.2f}",
            "states where the action differs": check["states_where_action_differs"],
            "mean time in system (s), simulated": f"{evaluations[name]['mean_waiting_time_s']:.2f}",
            "dropped arrivals per episode, simulated": f"{evaluations[name]['mean_dropped_arrivals']:.2f}",
        }
        for name, check in (
            (name, scheduling["policy_checks"][name]) for name in names
        )
    }
).rename_axis("quantity")
```

The exact rows come from the linear solves. The simulated rows average 2,000
seeded episodes of 300 periods from the empty system. Value iteration
required 2,435 sweeps on this high-discount problem. Its independently
recomputed Bellman residual is $9.851\times10^{-11}$, and evaluating its policy
by a linear solve reproduces its value function to within $10^{-8}$ in every
state. Prefill priority is worse than the optimal policy by at least 3.02 in
every state and disagrees with it in 8 states, all on the switching curve,
where prefill priority keeps filling the batch to six. Decode priority is worse
by at least 131.82 in every state. In the simulated episodes it drops 4.35
arrivals per episode and roughly doubles the time a request spends in the
system. Prefill priority drops slightly fewer arrivals than the optimal policy,
0.14 against 0.20 per episode. The optimal policy minimizes the total
discounted cost, not each term of it, and accepts a few more drops for less
time spent waiting in the batch.

The reduced scheduling MDP has the following transition parameters and
value-iteration certificate.

```{code-cell} python
:tags: [remove-input]

dp_row = pd.read_csv("artifacts/inference_serving/metrics_dp.csv").iloc[0]
pd.DataFrame(
    {
        "value": [
            f"{dp_row['bellman_residual']:.3e}",
            f"{int(dp_row['iterations']):,}",
            dp_row["profile_status"],
            f"{dp_row['arrival_probability']:.6f}",
            f"{dp_row['prefill_completion_probability']:.6f}",
            f"{dp_row['decode_completion_probability']:.6f}",
            f"{dp_row['decode_batch_efficiency_at_cap']:.3f}",
            f"{dp_row['prefill_energy_j']:.3f}",
            f"{dp_row['decode_energy_j']:.3f}",
            f"{dp_row['idle_energy_j']:.3f}",
        ]
    },
    index=[
        "Bellman residual",
        "value-iteration sweeps",
        "profile provenance",
        "arrival probability λ",
        "prefill completion probability q_p",
        "single-stream decode completion probability q_1",
        "per-stream decode speed at d = 6, relative to d = 1",
        "prefill energy (J)",
        "decode energy (J)",
        "idle energy (J)",
    ],
).rename_axis("quantity")
```

{download}`Download the dynamic-programming certificate (CSV) <artifacts/inference_serving/metrics_dp.csv>`

The certificate concerns the supplied 49-state transition matrix. The result
is exact for the stated finite MDP, not for vLLM. Aggregating request ages and
lengths removes distinctions that can affect head-of-line waiting and cache
release; all prompts and responses are treated as having the trace's mean
length. The calibrated transition kernel is stationary, the clock is fixed,
temperature is absent from the state, the batch is capped at six, and the
action set excludes mixed prefill-decode batches. Measurements from one L4
deployment calibrate the phase rates, batch slowdown, and powers; they do not
establish scheduling performance across L4 systems. The request-level replay
from the earlier chapters remains a model-audit tool, not part of the
optimality proof.

:::{dropdown} Inspect the scheduling MDP, value iteration, and the policy check
```{literalinclude} code/inference_control.py
:language: python
:start-at: def make_scheduling_mdp
:end-before: def _scheduling_clock_calibration
:linenos:
```

```{literalinclude} code/inference_control.py
:language: python
:start-at: def solve_scheduling_mdp
:end-before: def _sample_next_state
:linenos:
```

{download}`Download the complete inference-control implementation <code/inference_control.py>`
:::

(sec-newton-kantorovich-pi)=
## Newton-Kantorovich and Policy Iteration

Value iteration is successive approximation applied to the Bellman optimality equation. This section develops the other general tool for operator equations, the Newton-Kantorovich method, and shows that applying it to the same equation gives the policy iteration algorithm of {ref}`sec-policy-iteration`. This explains why policy iteration needs so few iterations, and why modified policy iteration, which solves the Newton system only approximately, still works.

### Newton-Kantorovich Method

The Newton-Kantorovich method is a generalization of Newton's method from finite dimensional vector spaces to infinite dimensional function spaces: rather than iterating in the space of vectors, we are iterating in the space of functions. 

Newton's method is often written as the familiar update:

$$
x_{k+1} = x_k - [DF(x_k)]^{-1} F(x_k),
$$
which makes it look as though the essence of the method is "take a derivative and invert it." But the real workhorse behind Newton's method (both in finite and infinite dimensions) is **linearization**.

At each step, the idea is to replace the nonlinear operator $F:X \to Y$ by a local surrogate model of the form

$$
F(x+h) \approx F(x) + Lh,
$$
where $L$ is a linear map capturing how small perturbations in the input propagate to changes in the output. This is a Taylor-like expansion in Banach spaces: the role of the derivative is precisely to provide the correct notion of such a linear operator.

To find a root of $F$, we impose the condition that the surrogate vanishes at the next iterate:

$$
0 = F(x+h) \approx F(x) + Lh.
$$
Solving this linear equation gives the increment $h$. In finite dimensions, $L$ is the Jacobian matrix; in Banach spaces, it must be the **Fréchet derivative**.

But what exactly is a Fréchet derivative in infinite dimensions? To understand this, we need to generalize the concept of derivative from finite-dimensional calculus. In infinite-dimensional spaces, there are several notions of differentiability, each with different strengths and requirements:

**1. Gâteaux (Directional) Derivative**

We say that the Gâteaux derivative of $F$ at $x$ in a specific direction $h$ is:

$$
F'(x; h) = \lim_{t \to 0} \frac{F(x + th) - F(x)}{t}
$$

This quantity measures how the function $F$ changes along the ray $x + th$. While this limit may exist for each direction $h$ separately, it doesn't guarantee that the derivative is linear in $h$. This is a key limitation: the Gâteaux derivative can exist in all directions but still fail to provide a good linear approximation.

**2. Hadamard Directional Derivative**

Rather than considering a single direction of perturbation, we now consider a bundle of perturbations around $h$. We ask how the function changes as we approach the target direction from nearby directions. We say that $F$ has a Hadamard directional derivative if:

$$
F'(x; h) = \lim_{\substack{t \downarrow 0 \\ h' \to h}} \frac{F(x + t h') - F(x)}{t}
$$

This is a stronger condition than Gâteaux differentiability because it requires the limit to be uniform over nearby directions. However, it still doesn't guarantee linearity in $h$.

**3. Fréchet Derivative**

The strongest and most natural notion: $F$ is Fréchet differentiable at $x$ if there exists a bounded linear operator $L$ such that:

$$
\lim_{h \to 0} \frac{\|F(x + h) - F(x) - Lh\|}{\|h\|} = 0
$$

This definition directly addresses the inadequacy of the previous notions. Unlike Gâteaux and Hadamard derivatives, the Fréchet derivative explicitly requires the existence of a linear operator $L$ that provides a good approximation. Key properties:

- $L$ must be **linear** in $h$ (unlike the directional derivatives above)
- The approximation error is $o(\|h\|)$, uniform in all directions
- This is the "true" derivative: it generalizes the Jacobian matrix to infinite dimensions
- Notation: $L = F'(x)$ or $DF(x)$

**Relationship:**

$$
\text{Fréchet differentiable} \Rightarrow \text{Hadamard directionally diff.} \Rightarrow \text{Gâteaux directionally diff.}
$$

In the context of the Newton-Kantorovich method, we work with an operator $F: X \to Y$ where both $X$ and $Y$ are Banach spaces. The Fréchet derivative $F'(x)$ is the best linear approximation of $F$ near $x$, and it's exactly this linear operator $L$ that we use in our linearization $F(x+h) \approx F(x) + F'(x)h$.

Now apart from those mathematical technicalities, Newton-Kantorovich has in essence the same structure as that of the original Newton's method. That is, it applies the following sequence of steps:

1. **Linearize the Operator**:
   Given an approximation $ x_n $, we consider the Fréchet derivative of $ F $, denoted by $ F'(x_n) $. This derivative is a linear operator that provides a local approximation of $ F $ near $ x_n $.

2. **Set Up the Newton Step**:
   The method then solves the linearized equation for a correction $ h_n $:

   $$
   F'(x_n) h_n = -F(x_n).
   $$
   This equation represents a linear system where $ h_n $ is chosen so that the linearized operator $ F(x_n) + F'(x_n)h_n $ equals zero.

3. **Update the Solution**:
   The new approximation $ x_{n+1} $ is then given by:

   $$
   x_{n+1} = x_n + h_n.
   $$
   This correction step refines $ x_n $, bringing it closer to the true solution.

4. **Repeat Until Convergence**:
   We repeat the linearization and update steps until the solution $ x_n $ converges to the desired tolerance, which can be verified by checking that $ \|F(x_n)\| $ is sufficiently small, or by monitoring the norm $ \|x_{n+1} - x_n\| $.

The convergence of Newton-Kantorovich does not hinge on $ F $ being a contraction over the entire domain (as it could be the case for successive approximation). The convergence properties of the Newton-Kantorovich method are as follows:

1. **Local Convergence**: Under mild conditions (e.g., $F$ is Fréchet differentiable and $F'(x)$ is invertible near the solution), the method converges locally. This means that if the initial guess is sufficiently close to the true solution, the method will converge.

2. **Global Convergence**: Global convergence is not guaranteed in general. However, under stronger conditions (e.g., $F$ is analytic and satisfies certain bounds), the method can converge globally.

3. **Rate of Convergence**: When the method converges, it typically exhibits quadratic convergence. This means that the error at each step is proportional to the square of the error at the previous step:

   $$
   \|x_{n+1} - x^*\| \leq C\|x_n - x^*\|^2
   $$

   where $x^*$ is the true solution and $C$ is some constant. This quadratic convergence is significantly faster than the linear convergence typically seen in methods like successive approximation.

### Newton-Kantorovich Applied to Bellman Optimality

We now apply the Newton-Kantorovich framework to the Bellman optimality equation. Let

$$
(\Bellman v)(s) = \max_{a \in A(s)} \left\{ r(s,a) + \gamma \sum_{s'} p(s' \mid s,a) v(s') \right\}.
$$

The problem is to find $v$ such that $\Bellman v = v$, or equivalently $\mathrm{B}(v) := \Bellman v - v = 0$. The operator $\Bellman$ is piecewise affine, hence not globally differentiable, but it is directionally differentiable everywhere in the Hadamard sense and Fréchet differentiable at points where the maximizer is unique.

We consider three complementary perspectives for understanding and computing its derivative.

#### Perspective 1: Max of Affine Maps

In tabular form, for finite state and action spaces, the Bellman operator can be written as a pointwise maximum of affine maps:

$$
(\Bellman v)(s) = \max_{a \in A(s)} \left\{ r(s,a) + \gamma (P_a v)(s) \right\},
$$
where $P_a \in \mathbb{R}^{|S| \times |S|}$ is the transition matrix associated with action $a$. Each $Q_a v := r^a + \gamma P_a v$ is affine in $v$. The operator $\Bellman$ therefore computes the upper envelope of a finite set of affine functions at each state.

At any $v$, let the **active set** at state $s$ be

$$
\mathcal{A}^*(s; v) := \arg\max_{a \in A(s)} (Q_a v)(s).
$$

Then the Hadamard directional derivative exists and is given by

$$
(\Bellman '(v; h))(s) = \max_{a \in \mathcal{A}^*(s; v)} \gamma (P_a h)(s).
$$

If the active set is a singleton, this expression becomes linear in $h$, and $\Bellman$ is Fréchet differentiable at $v$, with

$$
\Bellman'(v) = \gamma P_{\pi_v},
$$

where $\pi_v(s) := a^*(s)$ is the greedy policy at $v$. 
<!-- In the presence of ties, the derivative becomes set-valued: the Clarke subdifferential consists of stochastic matrices whose rows are convex combinations of the $\gamma P_a$ over $a \in \mathcal{A}^*(s; v)$. -->

#### Perspective 2: Envelope Theorem

Consider now a value function approximated as a linear combination of basis functions:

$$
v_c(s) = \sum_j c_j \phi_j(s).
$$

At a node $s_i$, define the parametric maximization

$$
v_i(c) := (\Bellman v_c)(s_i) = \max_{a \in A(s_i)} \left\{ r(s_i,a) + \gamma \sum_j c_j \mathbb{E}_{s' \mid s_i, a}[\phi_j(s')] \right\}.
$$

Define

$$
F_i(a, c) := r(s_i,a) + \gamma \sum_j c_j \mathbb{E}_{s' \mid s_i, a}[\phi_j(s')],
$$

so that $v_i(c) = \max_a F_i(a, c)$. Since $F_i$ is linear in $c$, we can apply the **envelope theorem** (Danskin's theorem): if the optimizer $a_i^*(c)$ is unique or selected measurably, then

$$
\frac{\partial v_i}{\partial c_j}(c) = \gamma \mathbb{E}_{s' \mid s_i, a_i^*(c)}[\phi_j(s')].
$$

We do not need to differentiate the optimizer $a_i^*(c)$ itself. The result extends to the subdifferential case when ties occur, where the Jacobian becomes set-valued.

This result is useful when solving the collocation equation $\Phi c = v(c)$. Newton's method requires the Jacobian $v'(c)$, and this expression allows us to compute it without involving any derivatives of the optimal action.

#### Perspective 3: The Implicit Function Theorem

The third perspective applies the implicit function theorem to understand when the Bellman operator is differentiable despite containing a max operator. The maximization problem defines an implicit relationship between the value function and the optimal action, and the implicit function theorem tells us when this relationship is smooth enough to differentiate through.

The Bellman operator is defined as

$$
(\Bellman v)(s) = \max_{a} \left\{ r(s,a) + \gamma \sum_j p(j \mid s,a) v(j) \right\}.
$$

The difficulty is that the max operator encodes a discrete selection: which action achieves the maximum. To apply the implicit function theorem, we reformulate this as follows. For each action $a$, define the **action-value function**:

$$
Q_a(v, s) := r(s,a) + \gamma \sum_j p(j \mid s,a) v(j).
$$

The optimal action at $v$ satisfies the **optimality condition**:

$$
Q_{a^*(s)}(v, s) \geq Q_a(v, s) \quad \text{for all } a.
$$

Now suppose that at a particular $v$, action $a^*(s)$ is a **strict local maximizer** in the sense that there exists $\delta > 0$ such that

$$
Q_{a^*(s)}(v, s) > Q_a(v, s) + \delta \quad \text{for all } a \neq a^*(s).
$$

This strict inequality is the regularity condition needed for the implicit function theorem. It ensures that the optimal action is unique at $v$ and remains so in a neighborhood of $v$.

To see why, consider any perturbation $v + h$ with $\|h\|$ small. Since $Q_a$ is linear in $v$, we have:

$$
Q_a(v+h, s) = Q_a(v, s) + \gamma \sum_j p(j \mid s,a) h(j).
$$

The perturbation term is bounded: $|\gamma \sum_j p(j \mid s,a) h(j)| \leq \gamma \|h\|$. Therefore, for $\|h\| < \delta/\gamma$, the strict gap ensures that

$$
Q_{a^*(s)}(v+h, s) > Q_a(v+h, s) \quad \text{for all } a \neq a^*(s).
$$

Thus $a^*(s)$ remains the unique maximizer throughout the neighborhood $\{v + h : \|h\| < \delta/\gamma\}$.

The implicit function theorem now applies: in this neighborhood, the mapping $v \mapsto a^*(s; v)$ is **constant** (and hence smooth), taking the value $a^*(s)$. This allows us to write

$$
(\Bellman v)(s) = Q_{a^*(s)}(v, s) = r(s,a^*(s)) + \gamma \sum_j p(j \mid s,a^*(s)) v(j)
$$
as an explicit formula that holds throughout the neighborhood. Since $Q_{a^*(s)}(\cdot, s)$ is an affine (hence smooth) function of $v$, we can differentiate it:

$$
\frac{d}{dv} (\Bellman v)(s) = \gamma P_{a^*(s)}.
$$

More precisely, for any perturbation $h$:

$$
(\Bellman (v+h))(s) = (\Bellman v)(s) + \gamma \sum_j p(j \mid s,a^*(s)) h(j) + o(\|h\|).
$$

This is the Fréchet derivative:

$$
\Bellman'(v) = \gamma P_{\pi_v},
$$

where $\pi_v(s) = a^*(s)$ is the greedy policy.

**The role of the implicit function theorem**: It guarantees that when the maximizer is unique with a strict gap (the regularity condition), the argmax function $v \mapsto a^*(s; v)$ is locally constant, which removes the non-differentiability of the max operator. Without this regularity condition (specifically, at points where multiple actions tie for optimality), the implicit function theorem does not apply, and the operator is not Fréchet differentiable. The active set perspective (Perspective 1) and the envelope theorem (Perspective 2) provide the tools to handle these non-smooth points.

### Connection to Policy Iteration

We return to the Newton-Kantorovich step:

$$
(I - \Bellman'(v_n)) h_n = v_n - \Bellman v_n,
\quad
v_{n+1} = v_n - h_n.
$$

Suppose $\Bellman'(v_n) = \gamma P_{\pi_{v_n}}$ for the greedy policy $\pi_{v_n}$. Then

$$
(I - \gamma P_{\pi_{v_n}}) v_{n+1} = r^{\pi_{v_n}},
$$

which is exactly policy evaluation for $\pi_{v_n}$. Recomputing the greedy policy from $v_{n+1}$ yields the next iterate.

Thus, **policy iteration is Newton-Kantorovich** applied to the Bellman optimality equation. At points of nondifferentiability (when ties occur), the operator is still semismooth, and policy iteration corresponds to a semismooth Newton method. The envelope theorem is what justifies the simplification of the Jacobian to $\gamma P_{\pi_v}$, bypassing the need to differentiate through the optimizer. This completes the equivalence.

### The Semismooth Newton Perspective

The three perspectives we developed above (the active set view, the envelope theorem, and the implicit function theorem) all point toward a deeper framework for understanding Newton-type methods on non-smooth operators. This framework, known as semismooth Newton methods, was developed precisely to handle operators like the Bellman operator that are piecewise smooth but not globally differentiable. The connection between policy iteration and semismooth Newton methods has been rigorously developed in recent work {cite}`Gargiani2022`.

The classical Newton-Kantorovich method assumes the operator is Fréchet differentiable everywhere. The derivative exists, is unique, and varies continuously with the base point. But the Bellman operator $\Bellman$ violates this assumption at any value function where multiple actions tie for optimality at some state. At such points, the implicit function theorem fails, and there is no unique Fréchet derivative. 

Semismooth Newton methods address this by replacing the notion of a single Jacobian with a generalized derivative that captures the behavior of the operator near non-smooth points. The most commonly used generalized derivative is the Clarke subdifferential, which we can think of as the convex hull of all possible "candidate Jacobians" that arise from limits approaching the non-smooth point from different directions.

For the Bellman residual $\mathrm{B}(v) = \Bellman v - v$, the Clarke subdifferential at a point $v$ can be characterized explicitly using our first perspective. Recall that at each state $s$, we defined the active set $\mathcal{A}^*(s; v) = \arg\max_a Q_a(v, s)$. When this set contains multiple actions, the operator is not Fréchet differentiable. However, it remains directionally differentiable in all directions, and the Clarke subdifferential consists of all matrices of the form

$$
\partial \mathrm{B}(v) = \left\{ I - \gamma P_\pi : \pi(s) \in \mathcal{A}^*(s; v) \text{ for all } s \right\}.
$$

In words, the generalized Jacobian is the set of all matrices $I - \gamma P_\pi$ where $\pi$ is any policy that selects an action from the active set at each state. When the maximizer is unique everywhere, this set reduces to a singleton, and we recover the classical Fréchet derivative. When ties occur, the set has multiple elements: precisely the convex combinations mentioned in Perspective 1.

The semismooth Newton method for solving $\mathrm{B}(v) = 0$ proceeds by selecting an element $J_k \in \partial \mathrm{B}(v_k)$ at each iteration and solving

$$
J_k h_k = -\mathrm{B}(v_k), \quad v_{k+1} = v_k + h_k.
$$

What this tells us is that any choice from the Clarke subdifferential yields a valid Newton-like update. In the context of the Bellman equation, choosing $J_k = I - \gamma P_{\pi_k}$ where $\pi_k$ is any greedy policy corresponds exactly to the policy evaluation step in policy iteration. The freedom in selecting which action to choose when ties occur translates to the freedom in selecting which element of the subdifferential to use.

Under appropriate regularity conditions (specifically, when the residual function is BD-regular or CD-regular), the semismooth Newton method converges locally at a quadratic rate {cite}`Gargiani2022`. This means that near the solution, the error decreases quadratically:
$$
\|v_{k+1} - v^*\| \leq C \|v_k - v^*\|^2.
$$

This theoretical result explains an empirical observation that has long been noted in practice: policy iteration typically converges in very few iterations, often just a handful, even when the state and action spaces are enormous and the space of possible policies is exponentially large. 

The semismooth Newton framework also suggests a spectrum of methods interpolating between value iteration and policy iteration. Value iteration can be interpreted as a Newton-like method where we choose $J_k = I$ at every iteration, ignoring the dependence of $\Bellman$ on $v$ entirely. This choice guarantees global convergence through the contraction property but sacrifices the quadratic local convergence rate. Policy iteration, at the other extreme, uses the full generalized Jacobian $J_k = I - \gamma P_{\pi_k}$, achieving quadratic convergence but at the cost of solving a linear system at each iteration.

Between these extremes lie methods that use approximate Jacobians or solve the Newton system only approximately. Modified policy iteration is one of them: it keeps the Jacobian $I - \gamma P_{\pi_k}$ but replaces its inverse by the first $m+1$ terms of the Neumann series. Another natural variant is to choose $J_k = \alpha I$ for some scalar $\alpha > 0$. This leads to the update

$$
v_{k+1} = \frac{\alpha - 1}{\alpha} v_k + \frac{1}{\alpha} \Bellman v_k.
$$

This is relaxed value iteration with step size $1/\alpha$. Taking $\alpha > 1$ damps the step: the update remains a global contraction, but with the larger modulus $1 - (1-\gamma)/\alpha$, so it is slower than standard value iteration. Taking $\alpha < 1$ lengthens the step, which is over-relaxation. The sup-norm argument then gives a contraction only for $\alpha > (1+\gamma)/2$, with modulus $(1+\gamma)/\alpha - 1$, which exceeds $\gamma$. Near the solution, however, the rate is set by the eigenvalues of $\gamma P_{\pi^*}$. The constant mode now contracts at $1 - (1-\gamma)/\alpha < \gamma$, so over-relaxation is faster whenever $P_{\pi^*}$ has no eigenvalues close to $-1$. For every $\alpha$, the update requires only pointwise operations rather than solving a linear system. The Newton perspective thus unifies existing algorithms and generates new ones by systematically exploring different approximations to the generalized Jacobian.

The connection to semismooth Newton methods places policy iteration within a broader mathematical framework that extends far beyond dynamic programming. Semismooth Newton methods are used in optimization (for complementarity problems and variational inequalities), in PDE-constrained optimization (for problems with control constraints), and in economics (for equilibrium problems). The Bellman equation, viewed through this lens, is simply one instance of a piecewise smooth equation, and the tools developed for such equations apply directly.

### Policy Iteration as a Newton Method

Comparing the Newton step above with {prf:ref}`policy-iteration-standard`, the linearization happens in the policy evaluation step 3, where we solve $(\mathbf{I}-\gamma \mathbf{P}_{\pi_n}) \mathbf{v} = \mathbf{r}_{\pi_n}$, and the improvement step recomputes the point at which the operator is linearized. The linear system can be solved by direct methods (Gaussian elimination), by successive approximation, or by Krylov methods such as GMRES. Solving it only approximately, with a few sweeps of successive approximation, is {prf:ref}`alg-modified-policy-iteration`: an inexact Newton method whose direction comes from a truncated Neumann series.

(sec-sailing-wind-shifts)=
## Example: Sailing to a Windward Mark in a Shifting Wind

How should a sailor beat to a mark upwind when the wind direction keeps
changing, and do the rules of thumb that racing sailors learn on the water
follow from the Bellman equation?

A sailboat cannot sail straight into the wind. To reach a mark that lies
upwind, it sails at an angle to the wind on one *tack*, with the wind coming
over one side of the boat, and then turns through the wind onto the other
tack. The boat is on *starboard* tack when the wind comes over its right side
and on *port* tack when it comes over its left side. Each change of tack
costs time. When the wind shifts, one tack points closer to the mark (it is
*lifted*) and the other points further away (it is *headed*). Racing sailors
are taught to tack when headed and to avoid sailing out to the *laylines*, the
lines from which the mark can be reached on a single tack, too early.
{cite:t}`Vanderbei1996Sailing` posed this problem as a stochastic dynamic
program on a grid. It later became the "sailing" benchmark for online
planning in {cite:t}`Peret2004`, {cite:t}`Kocsis2006` and
{cite:t}`Bonet2012`, where value iteration supplies the exact optimal values
against which sampled planners are scored.

**States.** The lake is a grid of $21\times 21$ waypoints. The state is
$s=(x,y,w,\tau)$: the boat's waypoint $(x,y)$, with $x$ pointing east and $y$
north; the direction $w\in\{\mathrm{N},\mathrm{NE},\ldots,\mathrm{NW}\}$ the
wind blows from; and the current tack $\tau\in\{\text{port},\text{starboard}\}$.
The tack must be part of the state because the cost of the next leg depends
on it. There are $21^2\times 8\times 2=7056$ states. The start is the waypoint
$(10,2)$ and the mark is $(10,18)$, 16 rows to the north. The mark is an
absorbing state with zero cost.

**Actions.** Each leg goes to one of the 8 neighbouring waypoints, so the
action is a heading $a\in\{\mathrm{N},\mathrm{NE},\ldots,\mathrm{NW}\}$.
Let $\theta(w,a)\in\{0,45,\ldots,180\}$ be the angle in degrees between the
wind direction and the heading. The heading $\theta=0$ points into the wind
and is not allowed, and neither is a heading that leaves the lake.

**Costs.** The cost of a leg is its duration in minutes. Following
{cite:t}`Vanderbei1996Sailing`, a leg between two neighbouring waypoints in a
straight line takes $t(45)=4$ minutes close-hauled, $t(90)=3$ minutes on a beam
reach and $t(135)=2$ minutes on a broad reach. Dead downwind takes
$t(180)=1$ minute, the default of the benchmark code of {cite:t}`Bonet2012`.
A diagonal leg is $\sqrt 2$ times longer, and we let it take $\sqrt 2$ times
as long, which reproduces the range of leg costs, 1 to 8.6 minutes, reported
by {cite:t}`Kocsis2006`. Let $\tau'(w,a,\tau)$ be the tack after the leg:
starboard if the wind comes from the right of the heading, port if it comes
from the left, and unchanged when sailing dead downwind. A change of tack
wastes $\delta=3$ minutes {cite:p}`Vanderbei1996Sailing`. The cost is

$$
c(s,a)=\ell(a)\,t\big(\theta(w,a)\big)+\delta\,\mathbb 1\{\tau'(w,a,\tau)\neq\tau\},
$$

where $\ell(a)=1$ for N, E, S, W and $\ell(a)=\sqrt2$ for the diagonals.

**Transitions.** The boat moves deterministically to the neighbouring
waypoint $(x',y')$ in direction $a$. At the end of the leg the wind keeps its
direction or shifts by $45^\circ$ according to a Markov chain $q(w'\mid w)$,
so

$$
p\big((x',y',w',\tau')\mid s,a\big)=q(w'\mid w).
$$

We compare three chains. In the *steady* chain the wind never changes. The
*oscillating* chain uses teaching values for a breeze that swings around
north: from north it shifts left or right with probability 0.1 each; from any
other direction it moves back towards north with probability 0.2, further
away with probability 0.1, and otherwise persists. The *shifty* chain is the
default matrix of the benchmark code of {cite:t}`Bonet2012`, rotated so that
the wind it favours blows from the north. It keeps the wind's direction with
probability 0.2 to 0.4 only.

**Objective.** The goal is the least expected time to the mark,

$$
v^*(s)=\min_\pi \mathbb E^\pi_s\Big[\sum_{t=0}^{T-1}c(S_t,A_t)\Big],
$$

where $T$ is the first leg that ends at the mark. This is a *stochastic
shortest-path* problem: there is no discount, and the horizon is random
because the race ends when the mark is reached. A discount factor would make
a minute late in the race cheaper than a minute at the start, which is not
what a racer wants, so we keep $\gamma=1$. The Bellman operator

$$
(\Bellman v)(s)=\min_{a\in\mathcal A_s}\Big\{c(s,a)+\sum_{w'}q(w'\mid w)\,v\big(x',y',w',\tau'(w,a,\tau)\big)\Big\}
$$

is then not a contraction in the sup norm, and the argument of
{prf:ref}`value-iteration-convergence` does not apply. A different argument
does. Some policy reaches the mark with probability one (a *proper* policy),
and every leg costs at least one minute, so a policy that fails to reach the
mark has infinite expected cost. Under these two conditions
{cite:t}`Bertsekas1991` show that $v^*$ is the unique fixed point of
$\Bellman$ and that value iteration converges to it from any starting
function. We run {prf:ref}`value-iteration` with $\min$ in place of $\max$
from $v_0=0$ and stop when $\|v_{n+1}-v_n\|_\infty<10^{-9}$ minutes. We also
run {prf:ref}`policy-iteration-standard` from a baseline policy described
below.

:::{iframe} ../interactive/sailing-wind-shifts.html
:label: fig-sailing-wind-shifts
:width: 100%
:title: Sailing in a shifting wind: the value-iteration policy, races against a layline sailor, and the value-iteration sweeps
:class: sailing-wind-shifts-replay
:placeholder: _static/sailing_wind_shifts/sailing.svg

The optimal policy for the oscillating breeze. (a) Wind from the north, boat
on starboard tack. Each arrow is the optimal heading from that waypoint; red
arrows change tack. Contours give the expected minutes to the mark (triangle)
and the dashed lines are the laylines of the mean wind. (b) The wind has
backed to the north-west, heading every starboard boat; almost everywhere the
policy tacks onto port, which now points straight north. (c) One race,
in which the optimal boat (blue) and the layline sailor (orange) see the same
wind on each leg; circles mark tacks. This race is a large win for the
optimal boat, 90.9 against 106.9 minutes, while the expected gap is 2.5
minutes. (d) Sup-norm error of
value iteration from $v_0=0$, its residual, and the error of policy iteration
started from the layline sailor. In the online book the replay races the two
boats leg by leg with the wind arrow and the arrow field for the current wind,
shows the distribution of finishing times over 4000 races, and animates the
value-iteration sweeps.
:::

The baseline is a *layline sailor* who ignores the shifts when deciding where
to tack. This sailor stays on starboard tack until reaching the port layline
of the mean (north) wind, then sails port tack towards the mark, and does the
mirror image on the right side of the course. On each leg the sailor takes
the allowed heading on that tack that ends closest to the mark. A second baseline,
the *certainty-equivalent* planner, replans every leg as if the wind it sees
now will never change. The table gives the expected time from the start with
the wind from the north. All values are exact policy evaluations. For the
optimal policy and the layline sailor, the mean of 4000 simulated races per
chain agrees with them within four standard errors.

:::{include} artifacts/sailing_wind_shifts/results.md
:::

In a steady wind the three policies coincide. The best course is one tack,
16 diagonal legs of $4\sqrt2$ minutes plus one 3-minute tack, or 93.5 minutes,
and the build script checks that value iteration returns exactly this
number. Shifts make the race shorter on average for every policy, because a
lifted boat can sail straight north at 4 minutes per row instead of
$4\sqrt2\approx5.7$. The optimal policy uses them best. In the oscillating
breeze it needs 84.1 minutes against 86.6 for the layline sailor; in the
shifty breeze, 72.4 against 75.2. The gain is about 3% of the race, and it is
a gain in expectation. Over 4000 paired races in the oscillating breeze, the
optimal boat finishes first in 62% of them and ties in 8%, so the layline
sailor still wins about three races in ten. The certainty-equivalent planner
sits in between, at 85.6 and 74.3 minutes, so reacting to the wind seen now
captures part of the gain and planning for the wind still to come captures
the rest.

Two pieces of sailing lore can be read off the policy. The first is *tack on
the headers*. Take the open water below the mark and at least three columns
inside both laylines, and look at every state in which the wind has shifted
$45^\circ$ from north, on either tack. In the oscillating breeze the optimal
policy ends the next leg on the lifted tack in all 900 such states; in the
shifty breeze it does so in 93% of them. A $45^\circ$ header leaves the headed
boat sailing across the course with no progress towards the mark, so paying 3
minutes to tack is worth it even though the shift may soon reverse. Panel (b)
of {numref}`fig-sailing-wind-shifts` shows this as a field of red arrows. The
second is *stay away from the laylines*. In the mean wind a starboard boat
eight rows below the mark tacks as soon as it is four columns left of the
mark, halfway to the layline (panel (a)). The reason is the one sailors
give. On the layline itself, any shift hurts: a header forces an extra tack, and a lift means the boat has sailed
further than needed. Staying near the middle of the course keeps both options
open. For a planner that assumes the wind seen now will persist, like the
certainty-equivalent planner, every tack point inside the laylines is equally
good, so nothing pushes it away from them. Neither rule is
written into the model; both come out of the minimization in $\Bellman$.

The two algorithms reach $v^*$ in different ways. Value iteration from
$v_0=0$ computes, at sweep $n$, the least expected cost of $n$ legs with no
penalty for where the boat ends. For a boat far from the mark, the cheapest
$n$ legs are cheap reaching and running legs that do not lead to the mark, so
$v_n$ stays well below $v^*$ there. The values settle near the mark
first, and the region within one minute of $v^*$ spreads outwards from the
mark until it covers the lake at sweep 37. The value-iteration mode of the
replay shows this. The sup-norm error stays above one minute for the first 36
sweeps and then collapses. In the oscillating breeze the residual reaches
exactly zero after 42 sweeps. Under the optimal policy every course reaches
the mark within 33 legs whatever the wind does, so value iteration ends like
backward induction with a horizon of a few dozen legs rather than converging
geometrically. Policy iteration started from the layline sailor needs 6
improvement steps. Each step evaluates a whole policy, the linear system of
{prf:ref}`policy-iteration-standard` (solved here by iterating that policy's
own Bellman operator), and the error falls from 14.4 minutes to zero.

The model is a teaching version of the real problem. Real wind shifts are
usually much smaller than $45^\circ$, and they happen in time, not at
waypoints; the boat's speed also depends on wind strength, which is fixed
here. The leg times and tack delay are the published values of the benchmark,
not measurements of any boat, and the oscillating chain, the lake and the two
baselines are our choices. No measured sailing data are used. What carries
over is the structure: a cost-to-go that depends on the wind as well as the
position, a value function that is the unique fixed point of the Bellman
operator without any discount, and sailing rules of thumb recovered as
properties of the optimal policy.

- {download}`Sailing model and solver <code/sailing_wind_shifts.py>`
- {download}`Sailing experiment builder <scripts/build_sailing_wind_shifts_artifacts.py>`
- {download}`Sailing results for the three wind chains <artifacts/sailing_wind_shifts/metrics.json>`

(sec-bridge-deck-maintenance)=
## Example: Maintaining a Bridge Deck

When should a highway agency repair a concrete bridge deck that has not failed
yet, and how much does a condition-based rule save over a fixed schedule?

Every highway bridge in the United States is inspected, usually every two
years, and the inspector gives the deck a condition rating from 9 (excellent)
to 0 (failed). The ratings are reported each year to the National Bridge
Inventory (NBI), where the deck rating is item 58. Decks get worse slowly and
at random, through cracking, corrosion of the reinforcing steel and spalling.
An agency can do nothing, seal and patch the deck, overlay or rehabilitate
it, or replace it. Work costs money now; a deck in poor condition costs
money later, through the risk of a failure that closes the bridge. The
trade-off repeats every year without a natural end, so we model it as an
infinite-horizon discounted MDP and solve it with the algorithms of this
chapter.

**States.** The state is the deck rating, observed once a year. Ratings 9 to 5
are kept apart and ratings 4 and below are merged into one *poor* state, so
$\mathcal S=\{9,8,7,6,5,{\le}4\}$. Merging the lowest ratings follows the deck
model of {cite:t}`Saifullah2024DDMAC`, whose costs we use below.

**Actions.** In every state $\mathcal A_s=\{\text{nothing},\text{minor},\text{major},\text{replace}\}$:
a minor repair (sealing and patching), a major repair (rehabilitation or an
overlay), or a new deck.

**Transitions.** The do-nothing kernel $p(j\mid s,\text{nothing})$ is
estimated from NBI records by counting. We use the deck histories assembled
by {cite:t}`Liu2023BridgeCNN`, which contain the item-58 rating of 88,596
decks for every year from 1993 to 2019 and are mirrored in their public code
repository. No history in that sample ever goes up, so it holds only decks
that received no rating-raising work. With $n_{sj}$ the number of years in
which a deck went from rating $s$ to rating $j$, the maximum-likelihood
estimate is

$$
\hat p(j\mid s,\text{nothing})=\frac{n_{sj}}{\sum_{k} n_{sk}} ,
$$

computed from 2,303,496 year-to-year pairs. The diagonal is large: a deck
stays at rating 9 with probability 0.85 per year, at 8 with 0.92, at 7 with
0.97, at 6 with 0.97 and at 5 with 0.98. These correspond to mean stays of
6.5, 12.9, 30.9, 36.4 and 55.7 years. No deck improves without work, so the
poor state is absorbing under "nothing".

The effects of work are teaching assumptions, in the spirit of the
component model of {cite:t}`Andriotis2020DRLInspection`. Let $g(s,a)$ be the
rating right after the work: a minor repair raises the rating by one but
cannot fix a poor deck, a major repair raises it by two up to at most 8, and
replacement gives a new deck, $g(s,\text{replace})=9$. The deck then
deteriorates for the rest of the year as if nothing had been done:

$$
p(j\mid s,a)=\hat p\big(j\mid g(s,a),\text{nothing}\big).
$$

**Costs.** Costs are in US dollars per square meter of deck. The work costs
are $k(\text{nothing})=0$, $k(\text{minor})=400$, $k(\text{major})=1{,}200$
and $k(\text{replace})=2{,}650$, from Table 7 of {cite:t}`Saifullah2024DDMAC`.
Their Table 6 gives an annual failure probability $q(s)$ of 0.001 for ratings
9 and 8, 0.005 for 7, 6 and 5, and 0.01 for the poor state, and they charge
a failure ten times the rebuilding cost. We take $F=10\times2{,}650=26{,}500$
and leave out the further cost they accrue while a failed bridge is out of
service. The one-period cost
is the work cost plus the expected failure cost of the repaired deck,

$$
c(s,a)=k(a)+q\big(g(s,a)\big)\,F .
$$

A deck left alone thus costs 26.5 per year at ratings 9 and 8, 132.5 at
ratings 7 to 5, and 265 when poor.

**Discount.** We use $\gamma=0.97$, a 3% annual rate, as
{cite:t}`Saifullah2024DDMAC` do.

The optimal value $v^\star(s)$ is the smallest expected discounted cost from
rating $s$, and it solves the Bellman equation with a minimum in place of the
maximum,

$$
v^\star(s)=\min_{a\in\mathcal A_s}\Big\{c(s,a)+\gamma\sum_{j\in\mathcal S}p(j\mid s,a)\,v^\star(j)\Big\}.
$$

We solve it twice. Value iteration ({prf:ref}`value-iteration`, with a
minimum) starts from $v_0=0$ and stops when
$\lVert v_{n+1}-v_n\rVert_\infty<\varepsilon(1-\gamma)/(2\gamma)$ with
$\varepsilon=10^{-6}$ dollars per square meter. With $\gamma=0.97$ the error
shrinks by about 3% per sweep, and the rule is met after 728 sweeps. Policy
iteration ({prf:ref}`policy-iteration-standard`) starts from "do nothing
everywhere", solves the $6\times6$ linear system
$(\mathbf I-\gamma\mathbf P_\pi)\mathbf v=\mathbf c_\pi$ at each step, and
stops after three evaluations. Its second policy already repairs every deck
rated 7 or worse; the third changes the action at two ratings and is optimal.
This is the Newton behavior described above: a handful of linear solves
instead of hundreds of contractions. The two methods return the same policy,
and the tests check it against all $4^6=4{,}096$ stationary deterministic
policies.

We compare the optimal policy with two rules that need no model. The
*reactive* rule does nothing until the deck is poor and then replaces it. The
*schedule* applies a minor repair every $k$ years whatever the rating, and
replaces the deck when it is poor; we report the best $k$ between 1 and 40.
Because the schedule depends on the year, we evaluate it exactly on the
Markov chain over the pair (year modulo $k$, rating).

:::{iframe} ../interactive/bridge-deck-maintenance.html
:label: fig-bridge-deck-maintenance
:width: 100%
:title: Bridge-deck maintenance: optimal policy against reactive and scheduled rules
:class: bridge-deck-maintenance-replay
:placeholder: _static/bridge_deck_maintenance/policy.svg

(a) Do-nothing transition probabilities between deck ratings, counted from
NBI records of 88,596 decks, 1993 to 2019; entries below 0.005 are left
blank. (b) Optimal action at each rating for failure costs from 2 to 40
times the replacement cost; the framed column is the base case. (c) Expected
discounted cost from each rating in the base case for the optimal policy, the
best schedule (a minor repair every 9 years) and the reactive rule. In the
online book the figure is a replay: two groups of 12 decks start from the
same ratings and use the same random numbers for 50 years, one under the
optimal policy and one under a baseline, with crews marking each repair and
counters of the discounted cost so far.
:::

In the base case the optimal policy does nothing at ratings 9 and 8, applies
a minor repair at ratings 7, 6 and 5, and a major repair when the deck is
poor. It never replaces a deck. The policy is a control limit: work starts
at the first rating where the failure risk jumps, from 26.5 to 132.5 dollars
per square meter per year, and the work gets heavier as the deck gets worse.
A minor repair at rating 7 costs 400 and moves the deck back to 8, which
saves about 106 per year for as long as it stays there, a mean of 12.9
years. For a new deck the expected discounted cost is 1,864 dollars per
square meter, against 2,951 for the best schedule and 3,200 for the reactive
rule.

:::{include} artifacts/bridge_deck_maintenance/results.md
:::

The table shows that the answer is driven by the price of failure. When a
failure costs only twice the replacement cost, the best policy is never to
touch the deck. The poor state then costs 53 per year, and no repair pays
for itself at a 3% discount rate. The control limit is not guaranteed
either. With a failure cost of five times replacement, the optimal policy
repairs decks rated 7 and 6 but leaves decks rated 5 alone. A minor repair
moves a deck from 5 to 6, where the failure probability is the same, so it
only delays the arrival in the poor state. With a failure cost of 40 times
replacement, a deck rated 6 gets a major repair and a deck rated 5 a minor
one, for the same reason: a major repair lifts 6 to 8, out of the
higher-risk band, while from 5 it would only reach 7. Monotone optimal
policies follow from conditions on the costs and transitions
{cite:p}`Puterman1994`, and the step-shaped failure table breaks them. In
every row the optimal policy beats both rules, and the gap grows with the
failure cost, to 3,720 dollars per square meter against the best schedule
and 10,344 against the reactive rule at 40 times replacement.

The replay shows the same comparison on sample paths. In the base case,
after 50 years the 12 decks under the optimal policy have cost 1,863 dollars
per square meter on average, in discounted terms, against 3,196 under the
reactive rule and 2,880 under the best schedule. The schedule wastes money
on decks that are still rated 9 or 8, while the reactive rule lets decks sit
at ratings 7 to 5, where the yearly risk is five times higher, for decades.

Several parts of this model are real and several are not. The do-nothing
kernel comes from real inspection records, but it is pooled over all decks
regardless of age, traffic, climate and material, and it is estimated from
a sample that excludes every deck whose rating went up. Decks that
deteriorated fast are the ones most likely to have been repaired, so the
mean stays above are probably too long. The work costs and failure
probabilities are published values that their authors call indicative, the
effects of the repairs are assumptions, and the cost has no term for traffic
delays or posting a weight limit. Inspections are treated as exact, while
real ratings are noisy, which turns the problem into the partially
observed one studied by {cite:t}`Andriotis2020DRLInspection`. Finally, an
agency manages thousands of decks under one budget, which couples the
per-deck problems that we solved separately here.

- {download}`Bridge-deck model and solver <code/bridge_deck_maintenance.py>`
- {download}`Bridge-deck experiment builder <scripts/build_bridge_deck_maintenance_artifacts.py>`
- {download}`NBI deck-rating transition counts <artifacts/bridge_deck_maintenance/deck_transition_counts.csv>`
- {download}`Bridge-deck results for every failure cost <artifacts/bridge_deck_maintenance/metrics.json>`

(sec-tamarisk-river)=
## Example: Controlling an Invasive Plant in a River Network

Which stretch of a river should be treated each year to keep an invasive
plant in check, and how large can the river get before an exact answer is
out of reach?

Tamarisk, or saltcedar, is a shrub that has invaded the banks of many rivers
in the western United States. It crowds out native trees such as willows and
cottonwoods, and its seeds travel with the water, so an infestation upstream
reseeds everything below it. Land managers can kill the plants in a stretch of
river, or kill them and plant natives in their place, but each treatment costs
money and only part of the network can be treated each year.
{cite:t}`Hall2012` formulated this problem as an MDP on a river network, and
{cite:t}`Taleghan2015` used it as a benchmark for planning with a simulator.
We build a small version of their model, solve it exactly with value
iteration and policy iteration, and then watch the exact methods run out of
room as the river grows.

**States.** The river is a binary tree of $n$ reaches, the stretches of
river between two confluences. Water flows from the leaves toward reach 1,
the outlet. Each reach has one site for a tree, as in the smallest instances
of {cite:t}`Taleghan2015`, and the site is empty ($E$), holds a native tree
($N$) or holds tamarisk ($T$). The state is
$s=(s_1,\ldots,s_n)\in\{E,N,T\}^n$, so $|\mathcal S|=3^n$.

**Actions.** For each reach $i$ the manager chooses $a_i\in\{0,1,2\}$: do
nothing, eradicate, or restore. Eradication kills a tamarisk plant with
probability $\chi=0.85$. Restoration first eradicates and then plants a native
tree in the empty site, which takes with probability $\beta=0.65$. Without a
budget the joint action $a=(a_1,\ldots,a_n)$ ranges over $3^n$ choices. With
the budget of {cite:t}`Taleghan2015`, at most one reach is treated per year,

$$
\mathcal A=\Big\{a\in\{0,1,2\}^n:\ \textstyle\sum_{i}\mathbf 1\{a_i\neq 0\}\le 1\Big\},
\qquad |\mathcal A| = 2n+1 .
$$

**Transitions.** A year has three steps. First the treatment acts on each
reach independently, with a $3\times 3$ kernel $\alpha_{a_i}(y_i\mid s_i)$.
Then every plant dies with probability $0.2$, with kernel
$\delta(z_i\mid y_i)$. Last, the survivors $z$ reseed the empty sites. Each
surviving plant releases 100 seeds, and a seed from reach $k$ lands in reach
$i$ with probability

$$
K_{ki}\ \propto\ 0.5^{\,d(k,i)}\;0.1^{\,u(k,i)},
$$

where $d(k,i)$ and $u(k,i)$ count the downstream and upstream steps on the
path from $k$ to $i$, and the weight that remains flows out of the outlet. The
values 0.5 and 0.1 are the downstream and upstream dispersal parameters of
{cite:t}`Taleghan2015`; the geometric form is our simplification of the model
they cite. Each empty reach also receives on average 1 tamarisk seed and 4
native seeds from outside the network. We treat the seeds that reach $i$ as
Poisson with means

$$
\lambda^T_i(z)=1+100\sum_{k:\,z_k=T}K_{ki},\qquad
\lambda^N_i(z)=4+100\sum_{k:\,z_k=N}K_{ki}.
$$

An empty site stays empty if no seed arrives. Otherwise one seed takes it,
and tamarisk wins with probability
$1.5\lambda^T_i/(1.5\lambda^T_i+\lambda^N_i)$, where 1.5 is one of the
values of tamarisk's competitive advantage examined by {cite:t}`Hall2012`. An occupied site
keeps its plant. Call $g_i(j_i\mid z)$ the resulting distribution of reach $i$
next year. Given the survivors, the reaches fill independently, so

$$
p(j\mid s,a)=\sum_{y,z}\ \prod_{i=1}^n\alpha_{a_i}(y_i\mid s_i)\
\prod_{i=1}^n\delta(z_i\mid y_i)\ \prod_{i=1}^n g_i(j_i\mid z),
\qquad
\mathbf P_a=\mathbf A_a\,\mathbf D\,\mathbf G ,
$$

with $\mathbf A_a=\alpha_{a_1}\otimes\cdots\otimes\alpha_{a_n}$ and
$\mathbf D=\delta\otimes\cdots\otimes\delta$ Kronecker products of $3\times 3$
matrices, and $\mathbf G$ a dense $3^n\times 3^n$ matrix. The coupling between
reaches enters only through $\mathbf G$.

**Costs and discount.** Following {cite:t}`Taleghan2015`, each invaded reach
costs 1.0 for the damage plus 0.1 for the tamarisk site it holds, and the
treatments cost $\kappa=(0,\,0.5,\,0.9)$ for nothing, eradicate and restore:

$$
c(s,a)=1.1\sum_{i=1}^n\mathbf 1\{s_i=T\}+\sum_{i=1}^n\kappa(a_i),
\qquad \gamma=0.9 .
$$

The costs are in abstract units. The goal is to minimize the expected
discounted cost $\mathbb E\big[\sum_{t\ge 0}\gamma^t c(S_t,A_t)\big]$.

**Solution.** Value iteration ({prf:ref}`value-iteration`, with a minimum in
place of the maximum) computes

$$
v_{k+1}(s)=\min_{a\in\mathcal A}\Big\{c(s,a)+\gamma\sum_{j}p(j\mid s,a)\,v_k(j)\Big\},
$$

and stops when $\|v_{k+1}-v_k\|_\infty<\varepsilon(1-\gamma)/(2\gamma)$ with
$\varepsilon=10^{-6}$. The factored form of $\mathbf P_a$ means that no
transition matrix is ever stored. Each sweep computes $\mathbf w=\mathbf D\mathbf G v_k$
once and then applies the small factors of $\mathbf A_a$ to $\mathbf w$ for
every action. Policy iteration ({prf:ref}`policy-iteration-standard`) starts
from "do nothing", solves $(\mathbf I-\gamma\mathbf P_\pi)v=\mathbf c_\pi$
exactly at each step, and improves greedily. On the 7-reach river, with
$3^7=2{,}187$ states and 15 actions, value iteration stops after 148 sweeps
and policy iteration after 4 improvements. The two value functions agree to
within $5\times10^{-7}$.

:::{iframe} ../interactive/tamarisk-river.html
:label: fig-tamarisk-river
:width: 100%
:title: Controlling tamarisk in a river network: the optimal policy against a heuristic
:class: tamarisk-river-replay
:placeholder: _static/tamarisk_river/river.svg

One simulated episode on a 7-reach river that starts fully invaded. Each
segment is a reach, colored by what grows there; water flows down to the
outlet. A ring marks the reach treated that year, R for restore and E for
eradicate. Top: the optimal policy computed by value iteration. Bottom: the
heuristic that restores the invaded reach farthest upstream. Both rows use the
same random numbers, and the titles give the discounted cost accumulated
before that year. In the online book the replay runs year by year, and the
reader can change the starting river, the comparison policy and the random
seed.
:::

Every policy in the table below is evaluated exactly, by solving its
policy-evaluation equations, so the numbers are expectations rather than
averages of simulated episodes.

:::{include} artifacts/tamarisk_river/results.md
:::

The optimal policy treats some invaded reach in every one of the 2,059
states that contain tamarisk, yet it agrees with the "most upstream first"
rule in only 678 of them and with the "most downstream first" rule in 672.
From a fully invaded river it restores reach 2, which sits between two
headwater reaches and the outlet, rather than a headwater reach or the outlet
itself. When a single tamarisk plant stands among natives, it eradicates
instead of restoring, because the native seed rain will usually refill the
site and the cheaper treatment suffices. These choices lower the expected cost
from a fully invaded river from 38.33 under the best heuristic to 36.89, and
from 14.53 to 13.30 averaged over all starting states. The budget matters
more than the choice of reach. Restoring every invaded reach each year, which
the budget forbids, costs 19.83 from a fully invaded river, and the optimal
policy without a budget costs 18.98.

The same solvers run on rivers of 1 to 8 reaches
({numref}`fig-tamarisk-river-scaling`). The number of value-iteration sweeps
barely changes, from 140 to 150, because it depends on $\gamma$ and
$\varepsilon$ rather than on the size of the model. Policy iteration needs 3
or 4 improvements at every size. What grows is the work per sweep. The states
grow as $3^n$, and a dense table of $p(j\mid s,a)$ would hold
$9^n(2n+1)$ numbers with the budget and $27^n$ without it. For 7 reaches
without a budget that is about $10^{10}$ numbers, or 84 GB. The factored
backup avoids this table, but the coupling matrix $\mathbf G$ alone has $9^n$
entries: 43 million for 8 reaches, and about $3.5\times10^9$, or 28 GB, for
10. On the machine that built the figures, the 8-reach model took about 10 s to
build and a few seconds to solve, and the 7-reach model without a budget took
several times as long to solve as with it. Timings vary from run to
run; the growth does not. The same blow-up hits the actions. The budget
keeps the joint action set at $2n+1$, but without it the minimization in each
backup ranges over $3^n$ joint actions.

:::{figure} _static/tamarisk_river/scaling.svg
:label: fig-tamarisk-river-scaling

Left: the number of states, the number of joint actions with the
one-treatment budget, and the number of entries in a dense table of
$p(j\mid s,a)$ without the budget, against the number of reaches $n$. Right:
seconds to build the model and solve it on one machine. Times below 0.1 s
are dominated by overhead.
:::

The model is a teaching version of the benchmark. The parameter values for
mortality, seed output, dispersal, treatment success, costs and the discount
factor come from {cite:t}`Taleghan2015`, and the competitive advantage from
{cite:t}`Hall2012`, but the river, the one-site reaches, the geometric
dispersal kernel, the Poisson seed rain and the order of events within a year
are our choices, and no field data enter the model. Real rivers have hundreds
of reaches and many sites per reach. {cite:t}`Taleghan2015` describe the full
dynamics as a dynamic Bayesian network too complex for exact inference, and
plan from a simulator instead. The example still
shows the point that the rest of the book builds on. Value iteration and
policy iteration converge in a number of steps that hardly depends on the
size of the river, but each step touches every state, and the number of
states is exponential in the number of reaches. Managing a real river needs
value functions that are represented compactly and backups that are
estimated from samples, which is the subject of
[approximate Bellman equations](approximate-bellman-equations.md) and
[fitted Q-iteration](fitted-q-iteration.md).

- {download}`Tamarisk river model and solvers <code/tamarisk_river.py>`
- {download}`Tamarisk river experiment builder <scripts/build_tamarisk_river_artifacts.py>`
- {download}`Tamarisk river results and timings <artifacts/tamarisk_river/metrics.json>`

## Summary and Outlook

Discounting converts an infinite stream of rewards into a bounded value and
makes the Bellman maps contractions in the sup norm. Value iteration applies
successive approximation; policy iteration alternates a fixed-policy linear
solve with greedy improvement, and modified policy iteration interpolates
between the two by truncating that solve. Value iteration's stopping rule comes
with a certificate of $\varepsilon$-optimality, and the span seminorm gives a
sharper one. The inference-scheduling example used that
linear solve as a check: no heuristic policy beat the value-iteration policy in
any of the 49 states. It also showed the price of state reduction: a smaller
MDP averages over distinctions that can still affect future completions.

The hard maximum in the Bellman operator selects one action and is
nondifferentiable at ties. Can the decision rule remain stochastic while the
operator becomes smooth? [Entropy-regularized dynamic programming](regularized-dp.md)
answers by placing a convex regularizer on the action distribution.

## Self-checks

:::{exercise} Contraction factor
:label: ex-dp-check-1

If two value functions differ by at most $\varepsilon$ in sup norm, by at most how much can their discounted Bellman updates differ?
:::

:::{solution} ex-dp-check-1
:class: dropdown

At most $\gamma\varepsilon$. The discounted Bellman operator is a $\gamma$-contraction in the sup norm.
:::

:::{exercise} Residual certificate
:label: ex-dp-check-2

Value iteration produces a Bellman residual $\|Tv-v\|_\infty=0.02$ with $\gamma=0.9$. Give the standard upper bound on $\|v-v^*\|_\infty$.
:::

:::{solution} ex-dp-check-2
:class: dropdown

$\|v-v^*\|_\infty\leq \|Tv-v\|_\infty/(1-\gamma)=0.02/0.1=0.2$.
:::

:::{exercise} Evaluation versus improvement
:label: ex-dp-check-3

Which step of policy iteration requires solving a linear fixed-policy problem, and which step takes an actionwise maximum?
:::

:::{solution} ex-dp-check-3
:class: dropdown

Policy evaluation solves $(I-\gamma P_\pi)v=r_\pi$. Policy improvement computes action values from that $v$ and chooses a greedy action in each state.
:::

:::{exercise} Reduced scheduling state
:label: ex-dp-check-4

Name one pair of request-level serving states that map to the same reduced
state $(p,d)$ but can have different future completion distributions.
:::

:::{solution} ex-dp-check-4
:class: dropdown

Two states can have the same numbers of waiting and decoding requests while
their active requests have different remaining output lengths. The reduced state discards those lengths, so its
transition kernel averages over them.
:::
