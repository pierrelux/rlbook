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

# Dynamic Programming

Rather than expressing the stochasticity in our system through a disturbance term as a parameter to a deterministic difference equation, we often work with an alternative representation (more common in operations research) which uses the Markov Decision Process formulation. The idea is that when we model our system in this way with the disturbance term being drawn indepently of the previous stages, the induced trajectory are those of a Markov chain. Hence, we can re-cast our control problem in that language, leading to the so-called Markov Decision Process framework in which we express the system dynamics in terms of transition probabilities rather than explicit state equations. In this framework, we express the probability that the system is in a given state using the transition probability function:

$$ p_t(\mathbf{x}_{t+1} | \mathbf{x}_t, \mathbf{u}_t) $$

This function gives the probability of transitioning to state $\mathbf{x}_{t+1}$ at time $t+1$, given that the system is in state $\mathbf{x}_t$ and action $\mathbf{u}_t$ is taken at time $t$. Therefore, $p_t$ specifies a conditional probability distribution over the next states: namely, the sum (for discrete state spaces) or integral over the next state should be 1.

Given the control theory formulation of our problem via a deterministic dynamics function and a noise term, we can derive the corresponding transition probability function through the following relationship:

$$
\begin{aligned}
p_t(\mathbf{x}_{t+1} | \mathbf{x}_t, \mathbf{u}_t) &= \mathbb{P}(\mathbf{W}_t \in \left\{\mathbf{w} \in \mathbf{W}: \mathbf{x}_{t+1} = f_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w})\right\}) \\
&= \sum_{\left\{\mathbf{w} \in \mathbf{W}: \mathbf{x}_{t+1} = f_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w})\right\}} q_t(\mathbf{w})
\end{aligned}
$$

Here, $q_t(\mathbf{w})$ represents the probability density or mass function of the disturbance $\mathbf{W}_t$ (assuming discrete state spaces). When dealing with continuous spaces, the above expression simply contains an integral rather than a summation. 


For a system with deterministic dynamics and no disturbance, the transition probabilities become much simpler and be expressed using the indicator function. Given a deterministic system with dynamics:

$$ \mathbf{x}_{t+1} = f_t(\mathbf{x}_t, \mathbf{u}_t) $$

The transition probability function can be expressed as:

$$ p_t(\mathbf{x}_{t+1} | \mathbf{x}_t, \mathbf{u}_t) = \begin{cases}
1 & \text{if } \mathbf{x}_{t+1} = f_t(\mathbf{x}_t, \mathbf{u}_t) \\
0 & \text{otherwise}
\end{cases} $$

With this transition probability function, we can recast our Bellman optimality equation:

$$ J_t^\star(\mathbf{x}_t) = \max_{\mathbf{u}_t \in \mathbf{U}} \left\{ c_t(\mathbf{x}_t, \mathbf{u}_t) + \sum_{\mathbf{x}_{t+1}} p_t(\mathbf{x}_{t+1} | \mathbf{x}_t, \mathbf{u}_t) J_{t+1}^\star(\mathbf{x}_{t+1}) \right\} $$

Here, ${c}(\mathbf{x}_t, \mathbf{u}_t)$ represents the expected immediate reward (or negative cost) when in state $\mathbf{x}_t$ and taking action $\mathbf{u}_t$ at time $t$. The summation term computes the expected optimal value for the future states, weighted by their transition probabilities.

This formulation offers several advantages:

1. It makes the Markovian nature of the problem explicit: the future state depends only on the current state and action, not on the history of states and actions.

2. For discrete-state problems, the entire system dynamics can be specified by a set of transition matrices, one for each possible action.

3. It allows us to bridge the gap with the wealth of methods in the field of probabilistic graphical models and statistical machine learning techniques for modelling and analysis. 

## Notation in Operations Reseach 

The presentation above was intended to bridge the gap between the control-theoretic perspective and the world of closed-loop control through the idea of determining the value function of a parametric optimal control problem. We then saw how the backward induction procedure was applicable to both the deterministic and stochastic cases by taking the expectation over the disturbance variable. We then said that we can alternatively work with a representation of our system where instead of writing our model as a deterministic dynamics function taking a disturbance as an input, we would rather work directly via its transition probability function, which gives rise to the Markov chain interpretation of our system in simulation.

Now we should highlight that the notation used in control theory tends to differ from that found in operations research communities, in which the field of dynamic programming flourished. We summarize those (purely notational) differences in this section.

In operations research, the system state at each decision epoch is typically denoted by $s \in \mathcal{S}$, where $S$ is the set of possible system states. When the system is in state $s$, the decision maker may choose an action $a$ from the set of allowable actions $\mathcal{A}_s$. The union of all action sets is denoted as $\mathcal{A} = \bigcup_{s \in \mathcal{S}} \mathcal{A}_s$.

The dynamics of the system are described by a transition probability function $p_t(j | s, a)$, which represents the probability of transitioning to state $j \in \mathcal{S}$ at time $t+1$, given that the system is in state $s$ at time $t$ and action $a \in \mathcal{A}_s$ is chosen. This transition probability function satisfies:

$$\sum_{j \in \mathcal{S}} p_t(j | s, a) = 1$$

It's worth noting that in operations research, we typically work with reward maximization rather than cost minimization, which is more common in control theory. However, we can easily switch between these perspectives by simply negating the quantity. That is, maximizing a reward function is equivalent to minimizing its negative, which we would then call a cost function.

The reward function is denoted by $r_t(s, a)$, representing the reward received at time $t$ when the system is in state $s$ and action $a$ is taken. In some cases, the reward may also depend on the next state, in which case it is denoted as $r_t(s, a, j)$. The expected reward can then be computed as:

$$r_t(s, a) = \sum_{j \in \mathcal{S}} r_t(s, a, j) p_t(j | s, a)$$

Combined together, these elemetns specify a Markov decision process, which is fully described by the tuple:

$$\{T, S, \mathcal{A}_s, p_t(\cdot | s, a), r_t(s, a)\}$$

where $\mathrm{T}$ represents the set of decision epochs (the horizon).

## Decision Rules and Policies

In the presentation provided so far, we directly assumed that the form of our feedback controller was of the form $u(\mathbf{x}, t)$. The idea is that rather than just looking at the stage as in open-loop control, we would now consider the current state to account for the presence of noise. We came to that conclusion by considering the parametric optimization problem corresponding to the trajectory optimization perspective and saw that the "argmax" counterpart to the value function (the max) was exactly this function $u(x, t)$. But this presentation was mostly for intuition and neglected the fact that we could consider other kinds of feedback controllers. In the context of MDPs and under the OR terminology, we should now rather talk of policies instead of controllers.

But to properly introduce the concept of policy, we first have to talk about decision rules. A decision rule is a prescription of a procedure for action selection in each state at a specified decision epoch. These rules can vary in their complexity due to their potential dependence on the history and ways in which actions are then selected. Decision rules can be classified based on two main criteria:

1. Dependence on history: Markovian or History-dependent
2. Action selection method: Deterministic or Randomized

Markovian decision rules are those that depend only on the current state, while history-dependent rules consider the entire sequence of past states and actions. Formally, we can define a history $h_t$ at time $t$ as:

$$h_t = (s_1, a_1, \ldots, s_{t-1}, a_{t-1}, s_t)$$

where $s_u$ and $a_u$ denote the state and action at decision epoch $u$. The set of all possible histories at time $t$, denoted $H_t$, grows rapidly with $t$:

$$H_1 = \mathcal{S}$$
$$H_2 = \mathcal{S} \times A \times \mathcal{S}$$
$$H_t = H_{t-1} \times A \times \mathcal{S} = \mathcal{S} \times (A \times \mathcal{S})^{t-1}$$

This exponential growth in the size of the history set motivates us to seek conditions under which we can avoid searching for history-dependent decision rules and instead focus on Markovian rules, which are much simpler to implement and evaluate.

Decision rules can be further classified as deterministic or randomized. A deterministic rule selects an action with certainty, while a randomized rule specifies a probability distribution over the action space.

These classifications lead to four types of decision rules:
1. Markovian Deterministic (MD): $d_t: \mathcal{S} \rightarrow \mathcal{A}_s$
2. Markovian Randomized (MR): $d_t: \mathcal{S} \rightarrow \mathcal{P}(\mathcal{A}_s)$
3. History-dependent Deterministic (HD): $d_t: H_t \rightarrow \mathcal{A}_s$
4. History-dependent Randomized (HR): $d_t: H_t \rightarrow \mathcal{P}(\mathcal{A}_s)$

Where $\mathcal{P}(\mathcal{A}_s)$ denotes the set of probability distributions over $\mathcal{A}_s$.

It's important to note that decision rules are stage-wise objects. However, to solve an MDP, we need a strategy for the entire horizon. This is where we make a distinction and introduce the concept of a policy. A policy $\pi$ is a sequence of decision rules, one for each decision epoch:

$$\pi = (d_1, d_2, ..., d_{N-1})$$

Where $N$ is the horizon length (possibly infinite). The set of all policies of class $K$ (where $K$ can be HR, HD, MR, or MD) is denoted as $\Pi^K$.

A special type of policy is a stationary policy, where the same decision rule is used at all epochs: $\pi = (d, d, ...)$, often denoted as $d^\infty$. 

The relationships between these policy classes form a hierarchy:

$$\begin{align*}
\Pi^{SD} \subset \Pi^{SR} \subset \Pi^{MR} \subset \Pi^{HR}\\
\Pi^{SD} \subset \Pi^{MD} \subset \Pi^{MR} \subset \Pi^{HR} \\
\Pi^{SD} \subset \Pi^{MD} \subset \Pi^{HD} \subset \Pi^{HR}
\end{align*}
$$

Where SD stands for Stationary Deterministic and SR for Stationary Randomized. The largest set is by far the set of history randomized policies. 

A fundamental question in MDP theory is: under what conditions can we avoid working with the set $\Pi^{HR}$ and focus for example on the much simpler set of deterministic Markovian policy? Even more so, we will see that in the infinite horizon case, we can drop the dependance on time and simply consider stationary deterministic Markovian policies. 

## What is an Optimal Policy?

Let's go back to the starting point and define what it means for a policy to be optimal in a Markov Decision Problem. For this, we will be considering different possible search spaces (policy classes) and compare policies based on the ordering of their value from any possible start state. The value of a policy $\pi$ (optimal or not) is defined as the expected total reward obtained by following that policy from a given starting state. Formally, for a finite-horizon MDP with $N$ decision epochs, we define the value function $v_\pi(s, t)$ as:

$$
v_\pi(s, t) \triangleq \mathbb{E}\left[\sum_{k=t}^{N-1} r_t(S_k, A_k) + r_N(S_N) \mid S_t = s\right]
$$

where $S_t$ is the state at time $t$, $A_t$ is the action taken at time $t$, and $r_t$ is the reward function. For simplicity, we write $v_\pi(s)$ to denote $v_\pi(s, 1)$, the value of following policy $\pi$ from state $s$ at the first stage over the entire horizon $N$.

In finite-horizon MDPs, our goal is to identify an optimal policy, denoted by $\pi^*$, that maximizes total expected reward over the horizon $N$. Specifically:

$$
v_{\pi^*}(s) \geq v_\pi(s), \quad \forall s \in \mathcal{S}, \quad \forall \pi \in \Pi^{\text{HR}}
$$

We call $\pi^*$ an **optimal policy** because it yields the highest possible value across all states and all policies within the policy class $\Pi^{\text{HR}}$. We denote by $v^*$ the maximum value achievable by any policy:

$$
v^*(s) = \max_{\pi \in \Pi^{\text{HR}}} v_\pi(s), \quad \forall s \in \mathcal{S}
$$

In reinforcement learning literature, $v^*$ is typically referred to as the "optimal value function," while in some operations research references, it might be called the "value of an MDP." An optimal policy $\pi^*$ is one for which its value function equals the optimal value function:

$$
v_{\pi^*}(s) = v^*(s), \quad \forall s \in \mathcal{S}
$$

It's important to note that this notion of optimality applies to every state. Policies optimal in this sense are sometimes called "uniformly optimal policies." A weaker notion of optimality, often encountered in reinforcement learning practice, is optimality with respect to an initial distribution of states. In this case, we seek a policy $\pi \in \Pi^{\text{HR}}$ that maximizes:

$$
\sum_{s \in \mathcal{S}} v_\pi(s) P_1(S_1 = s)
$$

where $P_1(S_1 = s)$ is the probability of starting in state $s$.

A fundamental result in MDP theory states that the maximum value can be achieved by searching over the space of deterministic Markovian Policies. Consequently:

$$ v^*(s) = \max_{\pi \in \Pi^{\mathrm{HR}}} v_\pi(s) = \max _{\pi \in \Pi^{M D}} v_\pi(s), \quad s \in S$$

This equality significantly simplifies the computational complexity of our algorithms, as the search problem can now be decomposed into $N$ sub-problems in which we only have to search over the set of possible actions. This is the backward induction algorithm, which we present a second time, but departing this time from the control-theoretic notation and using the MDP formalism:  

````{prf:algorithm} Backward Induction
:label: backward-induction

**Input:** State space $S$, Action space $A$, Transition probabilities $p_t$, Reward function $r_t$, Time horizon $N$

**Output:** Optimal value functions $v^*$

1. Initialize:
   - Set $t = N$
   - For all $s_N \in S$:

     $$v^*(s_N, N) = r_N(s_N)$$

2. For $t = N-1$ to $1$:
   - For each $s_t \in S$:
     a. Compute the optimal value function:

        $$v^*(s_t, t) = \max_{a \in A_{s_t}} \left\{r_t(s_t, a) + \sum_{j \in S} p_t(j | s_t, a) v^*(j, t+1)\right\}$$
     
     b. Determine the set of optimal actions:

        $$A_{s_t,t}^* = \arg\max_{a \in A_{s_t}} \left\{r_t(s_t, a) + \sum_{j \in S} p_t(j | s_t, a) v^*(j, t+1)\right\}$$

3. Return the optimal value functions $u_t^*$ and optimal action sets $A_{s_t,t}^*$ for all $t$ and $s_t$
````

Note that the same procedure can also be used for finding the value of a policy with minor changes; 

````{prf:algorithm} Policy Evaluation
:label: backward-policy-evaluation

**Input:** 
- State space $S$
- Action space $A$
- Transition probabilities $p_t$
- Reward function $r_t$
- Time horizon $N$
- A markovian deterministic policy $\pi$

**Output:** Value function $v^\pi$ for policy $\pi$

1. Initialize:
   - Set $t = N$
   - For all $s_N \in S$:

     $$v_\pi(s_N, N) = r_N(s_N)$$

2. For $t = N-1$ to $1$:
   - For each $s_t \in S$:
     a. Compute the value function for the given policy:

        $$v_\pi(s_t, t) = r_t(s_t, d_t(s_t)) + \sum_{j \in S} p_t(j | s_t, d_t(s_t)) v_\pi(j, t+1)$$

3. Return the value function $v^\pi(s_t, t)$ for all $t$ and $s_t$
````

This code could also finally be adapted to support randomized policies using:

$$v_\pi(s_t, t) = \sum_{a_t \in \mathcal{A}_{s_t}} d_t(a_t \mid s_t) \left( r_t(s_t, a_t) + \sum_{j \in S} p_t(j | s_t, a_t) v_\pi(j, t+1) \right)$$


## Example: Sample Size Determination in Pharmaceutical Development

Pharmaceutical development is the process of bringing a new drug from initial discovery to market availability. This process is lengthy, expensive, and risky, typically involving several stages:

1. **Drug Discovery**: Identifying a compound that could potentially treat a disease.
2. **Preclinical Testing**: Laboratory and animal testing to assess safety and efficacy.
. **Clinical Trials**: Testing the drug in humans, divided into phases:
   - Phase I: Testing for safety in a small group of healthy volunteers.
   - Phase II: Testing for efficacy and side effects in a larger group with the target condition.
   - Phase III: Large-scale testing to confirm efficacy and monitor side effects.
4. **Regulatory Review**: Submitting a New Drug Application (NDA) for approval.
5. **Post-Market Safety Monitoring**: Continuing to monitor the drug's effects after market release.

This process can take 10-15 years and cost over $1 billion {cite}`Adams2009`. The high costs and risks involved call for a principled approach to decision making. We'll focus on the clinical trial phases and NDA approval, per the MDP model presented by {cite}`Chang2010`:

1. **States** ($S$): Our state space is $S = \{s_1, s_2, s_3, s_4\}$, where:
   - $s_1$: Phase I clinical trial
   - $s_2$: Phase II clinical trial
   - $s_3$: Phase III clinical trial
   - $s_4$: NDA approval

2. **Actions** ($A$): At each state, the action is choosing the sample size $n_i$ for the corresponding clinical trial. The action space is $A = \{10, 11, ..., 1000\}$, representing possible sample sizes.

3. **Transition Probabilities** ($P$): The probability of moving from one state to the next depends on the chosen sample size and the inherent properties of the drug.
      We define:

   - $P(s_2|s_1, n_1) = p_{12}(n_1) = \sum_{i=0}^{\lfloor\eta_1 n_1\rfloor} \binom{n_1}{i} p_0^i (1-p_0)^{n_1-i}$
     where $p_0$ is the true toxicity rate and $\eta_1$ is the toxicity threshold for Phase I.
     
  - Of particular interest is the transition from Phase II to Phase III which we model as:

    $P(s_3|s_2, n_2) = p_{23}(n_2) = \Phi\left(\frac{\sqrt{n_2}}{2}\delta - z_{1-\eta_2}\right)$

    where $\Phi$ is the cumulative distribution function (CDF) of the standard normal distribution:

      $\Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x e^{-t^2/2} dt$

    This is giving us the probability that we would observe a treatment effect this large or larger if the null hypothesis (no treatment effect) were true. A higher probability indicates stronger evidence of a treatment effect, making it more likely that the drug will progress to Phase III.

    In this expression, $\delta$ is called the "normalized treatment effect". In clinical trials, we're often interested in the difference between the treatment and control groups. The "normalized" part means we've adjusted this difference for the variability in the data. Specifically $\delta = \frac{\mu_t - \mu_c}{\sigma}$ where $\mu_t$ is the mean outcome in the treatment group, $\mu_c$ is the mean outcome in the control group, and $\sigma$ is the standard deviation of the outcome. A larger $\delta$ indicates a stronger treatment effect.

    Furthermore, the term $z_{1-\eta_2}$ is the $(1-\eta_2)$-quantile of the standard normal distribution. In other words, it's the value where the probability of a standard normal random variable being greater than this value is $\eta_2$. For example, if $\eta_2 = 0.05$, then $z_{1-\eta_2} \approx 1.645$. A smaller $\eta_2$ makes the trial more conservative, requiring stronger evidence to proceed to Phase III.

    Finally, $n_2$ is the sample size for Phase II. The $\sqrt{n_2}$ term reflects that the precision of our estimate of the treatment effect improves with the square root of the sample size.

   - $P(s_4|s_3, n_3) = p_{34}(n_3) = \Phi\left(\frac{\sqrt{n_3}}{2}\delta - z_{1-\eta_3}\right)$
     where $\eta_3$ is the significance level for Phase III.

4. **Rewards** ($R$): The reward function captures the costs of running trials and the potential profit from a successful drug:

   - $r(s_i, n_i) = -c_i(n_i)$ for $i = 1, 2, 3$, where $c_i(n_i)$ is the cost of running a trial with sample size $n_i$.
   - $r(s_4) = g_4$, where $g_4$ is the expected profit from a successful drug.

5. **Discount Factor** ($\gamma$): We use a discount factor $0 < \gamma \leq 1$ to account for the time value of money and risk preferences.

```{code-cell} ipython3
:tags: [hide-input]
:load: code/sample_size_drug_dev_dp.py

```

# Infinite-Horizon MDPs

It often makes sense to model control problems over infinite horizons. We extend the previous setting and define the expected total reward of policy $\pi \in \Pi^{\mathrm{HR}}$, $v^\pi$ as:

$$
v^\pi(s) = \mathbb{E}\left[\sum_{t=1}^{\infty} r(S_t, A_t)\right]
$$

One drawback of this model is that we could easily encounter values that are $+\infty$ or $-\infty$, even in a setting as simple as a single-state MDP which loops back into itself and where the accrued reward is nonzero.

Therefore, it is often more convenient to work with an alternative formulation which guarantees the existence of a limit: the expected total discounted reward of policy $\pi \in \Pi^{\mathrm{HR}}$ is defined to be:

$$
v_\gamma^\pi(s) \equiv \lim_{N \rightarrow \infty} \mathbb{E}\left[\sum_{t=1}^N \gamma^{t-1} r(S_t, A_t)\right]
$$

for $0 \leq \gamma < 1$ and when $\max_{s \in \mathcal{S}} \max_{a \in \mathcal{A}_s}|r(s, a)| = R_{\max} < \infty$, in which case, $|v_\gamma^\pi(s)| \leq (1-\gamma)^{-1} R_{\max}$.


Finally, another possibility for the infinite-horizon setting is the so-called average reward or gain of policy $\pi \in \Pi^{\mathrm{HR}}$ defined as:

$$
g^\pi(s) \equiv \lim_{N \rightarrow \infty} \frac{1}{N} \mathbb{E}\left[\sum_{t=1}^N r(S_t, A_t)\right]
$$

We won't be working with this formulation in this course due to its inherent practical and theoretical complexities. 

Extending the previous notion of optimality from finite-horizon models, a policy $\pi^*$ is said to be discount optimal for a given $\gamma$ if: 

$$
v_\gamma^{\pi^*}(s) \geq v_\gamma^\pi(s) \quad \text { for each } s \in S \text { and all } \pi \in \Pi^{\mathrm{HR}}
$$

Furthermore, the value of a discounted MDP $v_\gamma^*(s)$, is defined by:

$$
v_\gamma^*(s) \equiv \max _{\pi \in \Pi^{\mathrm{HR}}} v_\gamma^\pi(s)
$$

More often, we refer to $v_\gamma$ by simply calling it the optimal value function. 

As for the finite-horizon setting, the infinite horizon discounted model does not require history-dependent policies, since for any $\pi \in \Pi^{HR}$ there exists a $\pi^{\prime} \in \Pi^{MR}$ with identical total discounted reward:
$$
v_\gamma^*(s) \equiv \max_{\pi \in \Pi^{HR}} v_\gamma^\pi(s)=\max_{\pi \in \Pi^{MR}} v_\gamma^\pi(s) .
$$

## Random Horizon Interpretation of Discounting
The use of discounting can be motivated both from a modeling perspective and as a means to ensure that the total reward remains bounded. From the modeling perspective, we can view discounting as a way to weight more or less importance on the immediate rewards vs. the long-term consequences. There is also another interpretation which stems from that of a finite horizon model but with an uncertain end time. More precisely:

Let $v_\nu^\pi(s)$ denote the expected total reward obtained by using policy $\pi$ when the horizon length $\nu$ is random. We define it by:

$$
v_\nu^\pi(s) \equiv \mathbb{E}_s^\pi\left[\mathbb{E}_\nu\left\{\sum_{t=1}^\nu r(S_t, A_t)\right\}\right]
$$


````{prf:theorem} Random horizon interpretation of discounting
:label: prop-5-3-1
Suppose that the horizon $\nu$ follows a geometric distribution with parameter $\gamma$, $0 \leq \gamma < 1$, independent of the policy such that 
$P(\nu=n) = (1-\gamma) \gamma^{n-1}, \, n=1,2, \ldots$, then $v_\nu^\pi(s) = v_\gamma^\pi(s)$ for all $s \in \mathcal{S}$ .
````

````{prf:proof}
See proposition 5.3.1 in {cite}`Puterman1994`

$$
v_\nu^\pi(s) = E_s^\pi \left\{\sum_{n=1}^{\infty} \sum_{t=1}^n r(X_t, Y_t)(1-\gamma) \gamma^{n-1}\right\}.
$$

Under the bounded reward assumption and $\gamma < 1$, the series converges and we can reverse the order of summation :

\begin{align*}
v_\nu^\pi(s) &= E_s^\pi \left\{\sum_{t=1}^{\infty} \sum_{n=t}^{\infty} r(S_t, A_t)(1-\gamma) \gamma^{n-1}\right\} \\
&= E_s^\pi \left\{\sum_{t=1}^{\infty} \gamma^{t-1} r(S_t, A_t)\right\} = v_\gamma^\pi(s)
\end{align*}

where the last line follows from the geometric series: 

$$
\sum_{n=1}^{\infty} \gamma^{n-1} = \frac{1}{1-\gamma}
$$

````


## Vector Representation in Markov Decision Processes

Let V be the set of bounded real-valued functions on a discrete state space S. This means any function $ f \in V $ satisfies the condition:

$$
\|f\| = \max_{s \in S} |f(s)| < \infty.
$$
where notation $ \|f\| $ represents the sup-norm (or $ \ell_\infty $-norm) of the function $ f $. 

When working with discrete state spaces, we can interpret elements of V as vectors and linear operators on V as matrices, allowing us to leverage tools from linear algebra. The sup-norm ($\ell_\infty$ norm) of matrix $\mathbf{H}$ is defined as:

$$
\|\mathbf{H}\| \equiv \max_{s \in S} \sum_{j \in S} |\mathbf{H}_{s,j}|
$$

where $\mathbf{H}_{s,j}$ represents the $(s, j)$-th component of the matrix $\mathbf{H}$.

For a Markovian decision rule $d \in D^{MD}$, we define:

\begin{align*}
\mathbf{r}_d(s) &\equiv r(s, d(s)), \quad \mathbf{r}_d \in \mathbb{R}^{|S|}, \\
[\mathbf{P}_d]_{s,j} &\equiv p(j \mid s, d(s)), \quad \mathbf{P}_d \in \mathbb{R}^{|S| \times |S|}.
\end{align*}

For a randomized decision rule $d \in D^{MR}$, these definitions extend to:

\begin{align*}
\mathbf{r}_d(s) &\equiv \sum_{a \in A_s} d(a \mid s) \, r(s, a), \\
[\mathbf{P}_d]_{s,j} &\equiv \sum_{a \in A_s} d(a \mid s) \, p(j \mid s, a).
\end{align*}

In both cases, $\mathbf{r}_d$ denotes a reward vector in $\mathbb{R}^{|S|}$, with each component $\mathbf{r}_d(s)$ representing the reward associated with state $s$. Similarly, $\mathbf{P}_d$ is a transition probability matrix in $\mathbb{R}^{|S| \times |S|}$, capturing the transition probabilities under decision rule $d$.

For a nonstationary Markovian policy $\pi = (d_1, d_2, \ldots) \in \Pi^{MR}$, the expected total discounted reward is given by:

$$
\mathbf{v}_\gamma^{\pi}(s)=\mathbb{E}\left[\sum_{t=1}^{\infty} \gamma^{t-1} r\left(S_t, A_t\right) \,\middle|\, S_1 = s\right].
$$

Using vector notation, this can be expressed as:

$$
\begin{aligned}
\mathbf{v}_\gamma^{\pi} &= \sum_{t=1}^{\infty} \gamma^{t-1} \mathbf{P}_\pi^{t-1} \mathbf{r}_{d_1} \\
&= \mathbf{r}_{d_1} + \gamma \mathbf{P}_{d_1} \mathbf{r}_{d_2} + \gamma^2 \mathbf{P}_{d_1} \mathbf{P}_{d_2} \mathbf{r}_{d_3} + \cdots \\
&= \mathbf{r}_{d_1} + \gamma \mathbf{P}_{d_1} \left( \mathbf{r}_{d_2} + \gamma \mathbf{P}_{d_2} \mathbf{r}_{d_3} + \gamma^2 \mathbf{P}_{d_2} \mathbf{P}_{d_3} \mathbf{r}_{d_4} + \cdots \right).
\end{aligned}
$$

This formulation leads to a recursive relationship:

$$
\begin{align*}
\mathbf{v}_\gamma^\pi &= \mathbf{r}_{d_1} + \gamma \mathbf{P}_{d_1} \mathbf{v}_\gamma^{\pi^{\prime}}\\
&=\sum_{t=1}^{\infty} \gamma^{t-1} \mathbf{P}_\pi^{t-1} \mathbf{r}_{d_t}
\end{align*}
$$

where $\pi^{\prime} = (d_2, d_3, \ldots)$.


For stationary policies, where $\pi = d^{\infty} \equiv (d, d, \ldots)$, the total expected reward simplifies to:

$$
\begin{align*}
\mathbf{v}_\gamma^{d^\infty} &= \mathbf{r}_d+ \gamma \mathbf{P}_d \mathbf{v}_\gamma^{d^\infty} \\
&=\sum_{t=1}^{\infty} \gamma^{t-1} \mathbf{P}_d^{t-1} \mathbf{r}_{d}
\end{align*}
$$

This last expression is called a Neumann series expansion, and it's guaranteed to exists under the assumptions of bounded reward and discount factor strictly less than one. Consequently, for a stationary policy, $\mathbf{v}_\gamma^{d^\infty}$ can be determined as the solution to the linear equation:

$$
\mathbf{v} = \mathbf{r}_d+ \gamma \mathbf{P}_d\mathbf{v},
$$

which can be rearranged to:

$$
(\mathbf{I} - \gamma \mathbf{P}_d) \mathbf{v} = \mathbf{r}_d.
$$

We can also characterize $\mathbf{v}_\gamma^{d^\infty}$ as the solution to an operator equation. More specifically, define the linear transformation $\mathrm{L}_d$ by

$$
\mathrm{L}_d \mathbf{v} \equiv \mathbf{r}_d+\gamma \mathbf{P}_dv
$$

for any $v \in V$. Intuitively, $\mathrm{L}_d$ takes a value function $\mathbf{v}$ as input and returns a new value function that combines immediate rewards ($\mathbf{r}_d$) with discounted future values ($\gamma \mathbf{P}_d\mathbf{v}$). 
<!-- 

A key property of $\mathrm{L}_d$ is that it maps bounded functions to bounded functions, provided certain conditions are met. This is formalized in the following lemma:


```{prf:lemma} Bounded Reward and Value Function
:label: bounded-reward-value

Let $S$ be a discrete state space, and assume that the reward function is bounded such that $|r(s, a)| \leq M$ for all actions $a \in A_s$ and states $s \in S$. For any discount factor $\gamma$ where $0 \leq \gamma \leq 1$, and for all value functions $v \in V$ and decision rules $d \in D^{MR}$, the following holds:

$$
\mathbf{r}_d+ \gamma \mathbf{P}_d \mathbf{v} \in V
$$

where $V$ is the space of bounded real-valued functions on $S$, $\mathbf{r}_d$ is the reward vector, and $\mathbf{P}_d$ is the transition probability matrix under decision rule $d$.
```

```{prf:proof}
We will prove this lemma in three steps:

1. First, we'll show that $\mathbf{r}_d\in V$.
2. Then, we'll prove that $\mathbf{P}_dv \in V$ for all $v \in V$.
3. Finally, we'll combine these results to show that $\mathbf{r}_d+ \gamma \mathbf{P}_dv \in V$.

**Step 1: Showing $\mathbf{r}_d\in V$**

Given that $|r(s, a)| \leq M$ for all $a \in A_s$ and $s \in S$, we can conclude that $\|\mathbf{r}_d\| \leq M$ for all $d \in D^{MR}$. This is because $\mathbf{r}_d$ is a vector whose components are weighted averages of $r(s, a)$ values, which are all bounded by $M$. Therefore, $\mathbf{r}_d$ is a bounded function on $S$, meaning $\mathbf{r}_d\in V$.

**Step 2: Showing $\mathbf{P}_d \mathbf{v} \in V$ for all $\mathbf{v} \in V$**

$\mathbf{P}_d$ is a probability matrix, which means that each row sums to 1. This property implies that $\|\mathbf{P}_d\| = 1$. Now, for any $v \in V$, we have:

$$
\|\mathbf{P}_d\mathbf{v}\| \leq \|\mathbf{P}_d\| \|\mathbf{v}\| = \|\mathbf{v}\|
$$

This inequality shows that if $v$ is bounded (which it is, since $v \in V$), then $\mathbf{P}_dv$ is also bounded by the same value. Therefore, $\mathbf{P}_d\mathbf{v} \in V$ for all $\mathbf{v} \in V$.

**Step 3: Combining the results**

We've shown that $\mathbf{r}_d\in V$ and $\mathbf{P}_d \mathbf{v} \in V$. Since $V$ is a vector space, it is closed under addition and scalar multiplication. Therefore, for any $\gamma$ where $0 \leq \gamma \leq 1$, we can conclude that:

$$
\mathbf{r}_d+ \gamma \mathbf{P}_d \mathbf{v} \in V
$$

This completes the proof.
``` -->
Therefore, we view $\mathrm{L}_d$ as an operator mapping elements of $V$ to $V$: ie. $\mathrm{L}_d: V \rightarrow V$. The fact that the value function of a policy is the solution to a system of equations can then be expressed with the statement: 

$$
\mathbf{v}_\gamma^{d^\infty}=\mathrm{L}_d \mathbf{v}_\gamma^{d^\infty} \text {. }
$$

## Solving Operator Equations

The operator equation we encountered in MDPs, $\mathbf{v}_\gamma^{d^\infty} = \mathrm{L}_d \mathbf{v}_\gamma^{d^\infty}$, is a specific instance of a more general class of problems known as operator equations. These equations appear in various fields of mathematics and applied sciences, ranging from differential equations to functional analysis.

Operator equations can take several forms, each with its own characteristics and solution methods:

1. **Fixed Point Form**: $x = \mathrm{T}(x)$, where $\mathrm{T}: X \rightarrow X$.
   Common in fixed-point problems, such as our MDP equation, we seek a fixed point $x^*$ such that $x^* = \mathrm{T}(x^*)$.

2. **General Operator Equation**: $\mathrm{T}(x) = y$, where $\mathrm{T}: X \rightarrow Y$.
   Here, $X$ and $Y$ can be different spaces. We seek an $x \in X$ that satisfies the equation for a given $y \in Y$.

3. **Nonlinear Equation**: $\mathrm{T}(x) = 0$, where $\mathrm{T}: X \rightarrow Y$.
   A special case of the general operator equation where we seek roots or zeros of the operator.

4. **Variational Inequality**: Find $x^* \in K$ such that $\langle \mathrm{T}(x^*), x - x^* \rangle \geq 0$ for all $x \in K$.
   Here, $K$ is a closed convex subset of $X$, and $\mathrm{T}: K \rightarrow X^*$ (the dual space of $X$). These problems often arise in optimization, game theory, and partial differential equations.

### Successive Approximation Method

For equations in fixed point form, a common numerical solution method is successive approximation, also known as fixed-point iteration:

````{prf:algorithm} Successive Approximation
:label: successive-approximation

**Input:** An operator $\mathrm{T}: X \rightarrow X$, an initial guess $x_0 \in X$, and a tolerance $\epsilon > 0$  
**Output:** An approximate fixed point $x^*$ such that $\|x^* - \mathrm{T}(x^*)\| < \epsilon$

1. Initialize $n = 0$  
2. **repeat**  
    3. Compute $x_{n+1} = \mathrm{T}(x_n)$  
    4. If $\|x_{n+1} - x_n\| < \epsilon$, **return** $x_{n+1}$  
    5. Set $n = n + 1$  
6. **until** convergence or maximum iterations reached  

````

The convergence of successive approximation depends on the properties of the operator $\mathrm{T}$. In the simplest and most common setting, we assume $\mathrm{T}$ is a contraction mapping. The Banach Fixed-Point Theorem then guarantees that $\mathrm{T}$ has a unique fixed point, and the successive approximation method will converge to this fixed point from any starting point. Specifically, $\mathrm{T}$ is a contraction if there exists a constant $q \in [0,1)$ such that for all $x,y \in X$:

$$
d(\mathrm{T}(x), \mathrm{T}(y)) \leq q \cdot d(x,y)
$$

where $d$ is the metric on $X$. In this case, the rate of convergence is linear, with error bound:

$$
d(x_n, x^*) \leq \frac{q^n}{1-q} d(x_1, x_0)
$$

However, the contraction mapping condition is not the only one that can lead to convergence. For instance, if $\mathrm{T}$ is nonexpansive (i.e., Lipschitz continuous with Lipschitz constant 1) and $X$ is a Banach space with certain geometrical properties (e.g., uniformly convex), then under additional conditions (e.g., $\mathrm{T}$ has at least one fixed point), the successive approximation method can still converge, albeit potentially more slowly than in the contraction case.

In practice, when dealing with specific problems like MDPs or differential equations, the properties of the operator often naturally align with one of these convergence conditions. For example, in discounted MDPs, the Bellman operator is a contraction in the supremum norm, which guarantees the convergence of value iteration.

### Newton-Kantorovich Method

The Newton-Kantorovich method is a generalization of Newton's method from finite dimensional vector spaces to infinite dimensional function spaces: rather than iterating in the space of vectors, we are iterating in the space of functions. Just as in the finite-dimensional counterpart, the idea is to improve the rate of convergence of our method by taking an "educated guess" on where to move next using a linearization of our operator at the current point. Now the concept of linearization, which is synonymous with derivative, will also require a generalization. Here we are in essence trying to quantify how the output of the operator $\mathrm{T}$ -- a function -- varies as we perturb its input -- also a function. The right generalization here is that of the Fréchet derivative.

Before we delve into the Fréchet derivative, it's important to understand the context in which it operates: Banach spaces. A Banach space is a complete normed vector space: ie a vector space, that has a norm, and which every Cauchy sequence convergeces.   Banach spaces provide a natural generalization of finite-dimensional vector spaces to infinite-dimensional settings. 
The norm in a Banach space allows us to quantify the "size" of functions and the "distance" between functions. This allows us to define notions of continuity, differentiability, and for analyzing the convergence of our method.
Furthermore, the completeness property of Banach spaces ensures that we have a well-defined notion of convergence. 

In the context of the Newton-Kantorovich method, we typically work with an operator $\mathrm{T}: X \to Y$, where both $X$ and $Y$ are Banach spaces and whose Fréchet derivative at a point $x \in X$, denoted $\mathrm{T}'(x)$, is a bounded linear operator from $X$ to $Y$ such that:

$$
\lim_{h \to 0} \frac{\|\mathrm{T}(x + h) - \mathrm{T}(x) - \mathrm{T}'(x)h\|_Y}{\|h\|_X} = 0
$$

where $\|\cdot\|_X$ and $\|\cdot\|_Y$ are the norms in $X$ and $Y$ respectively. In other words, $\mathrm{T}'(x)$ is the best linear approximation of $\mathrm{T}$ near $x$.

Now apart from those mathematical technicalities, Newton-Kantorovich has in essence the same structure as that of the original Newton's method. That is, it applies the following sequence of steps:

1. **Linearize the Operator**:
   Given an approximation $ x_n $, we consider the Fréchet derivative of $ \mathrm{T} $, denoted by $ \mathrm{T}'(x_n) $. This derivative is a linear operator that provides a local approximation of  $ \mathrm{T} $, near $ x_n $.

2. **Set Up the Newton Step**:
   The method then solves the linearized equation for a correction $ h_n $:

   $$
   \mathrm{T}(x_n) h_n = \mathrm{T}(x_n) - x_n.
   $$
   This equation represents a linear system where $ h_n $ is chosen to minimize the difference between $ x_n $ and $ \mathrm{T}(x_n) $ with respect to the operator's local behavior.

3. **Update the Solution**:
   The new approximation $ x_{n+1} $ is then given by:

   $$
   x_{n+1} = x_n - h_n.
   $$
   This correction step refines $ x_n $, bringing it closer to the true solution.

4. **Repeat Until Convergence**:
   We repeat the linearization and update steps until the solution $ x_n $ converges to the desired tolerance, which can be verified by checking that $ \|\mathrm{T}(x_n) - x_n\| $ is sufficiently small, or by monitoring the norm $ \|x_{n+1} - x_n\| $.

The convergence of Newton-Kantorovich does not hinge on $ \mathrm{T} $ being a contraction over the entire domain -- as it could be the case for successive approximation. The convergence properties of the Newton-Kantorovich method are as follows:

1. **Local Convergence**: Under mild conditions (e.g., $\mathrm{T}$ is Fréchet differentiable and $\mathrm{T}'(x)$ is invertible near the solution), the method converges locally. This means that if the initial guess is sufficiently close to the true solution, the method will converge.

2. **Global Convergence**: Global convergence is not guaranteed in general. However, under stronger conditions (e.g.,  $ \mathrm{T} $, is analytic and satisfies certain bounds), the method can converge globally.

3. **Rate of Convergence**: When the method converges, it typically exhibits quadratic convergence. This means that the error at each step is proportional to the square of the error at the previous step:

   $$
   \|x_{n+1} - x^*\| \leq C\|x_n - x^*\|^2
   $$

   where $x^*$ is the true solution and $C$ is some constant. This quadratic convergence is significantly faster than the linear convergence typically seen in methods like successive approximation.

## Optimality Equations for Infinite-Horizon MDPs

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

We can adopt an operator-theoretic perspective by defining a nonlinear operator $\mathrm{L}$ on the space $V$ of bounded real-valued functions on the state space $S$. We can then show that the value of an MDP is the solution to the following fixed-point equation:

$$
\mathrm{L} \mathbf{v} = \max_{d \in D^{MD}} \left\{\mathbf{r}_d + \gamma \mathbf{P}_d \mathbf{v}\right\}
$$

where $D^{MD}$ is the set of Markov deterministic decision rules, $\mathbf{r}_d$ is the reward vector under decision rule $d$, and $\mathbf{P}_d$ is the transition probability matrix under decision rule $d$.

Note that while we write $\max_{d \in D^{MD}}$, we do not implement the above operator in this way. Written in this fashion, it would indeed imply that we first need to enumerate all Markov deterministic decision rules and pick the maximum. Now the fact that we compare policies based on their value functions in a componentwise fashion, maxing over the space of Markovian deterministic rules amounts to the following update in component form:

$$
(\mathrm{L} \mathbf{v})(s) = \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right\}
$$

The equivalence between these two forms can be shown mathematically, as demonstrated in the following proposition and proof.

```{prf:proposition}
The operator $\mathrm{L}$ defined as a maximization over Markov deterministic decision rules:

$$(\mathrm{L} \mathbf{v})(s) = \max_{d \in D^{MD}} \left\{r(s,d(s)) + \gamma \sum_{j \in \mathcal{S}} p(j|s,d(s)) v(j)\right\}$$

is equivalent to the componentwise maximization over actions:

$$(\mathrm{L} \mathbf{v})(s) = \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right\}$$
```

```{prf:proof}
Let's define the right-hand side of the first equation as $R_1$ and the right-hand side of the second equation as $R_2$. We'll prove that $R_1 \leq R_2$ and $R_2 \leq R_1$, which will establish their equality.

Step 1: Proving $R_1 \leq R_2$

For any $d \in D^{MD}$, we have:

$$r(s,d(s)) + \gamma \sum_{j \in \mathcal{S}} p(j|s,d(s)) v(j) \leq \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right\}$$

This is because $d(s) \in \mathcal{A}_s$, so the left-hand side is included in the set over which we're maximizing on the right-hand side.

Taking the maximum over all $d \in D^{MD}$ on the left-hand side doesn't change this inequality:

$$\max_{d \in D^{MD}} \left\{r(s,d(s)) + \gamma \sum_{j \in \mathcal{S}} p(j|s,d(s)) v(j)\right\} \leq \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right\}$$

Therefore, $R_1 \leq R_2$.

Step 2: Proving $R_2 \leq R_1$

Let $a^* \in \mathcal{A}_s$ be the action that achieves the maximum in $R_2$. We can construct a Markov deterministic decision rule $d^*$ such that $d^*(s) = a^*$ and $d^*(s')$ is arbitrary for $s' \neq s$. Then:

$$\begin{align*}
R_2 &= r(s,a^*) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a^*) v(j) \\
&= r(s,d^*(s)) + \gamma \sum_{j \in \mathcal{S}} p(j|s,d^*(s)) v(j) \\
&\leq \max_{d \in D^{MD}} \left\{r(s,d(s)) + \gamma \sum_{j \in \mathcal{S}} p(j|s,d(s)) v(j)\right\} \\
&= R_1
\end{align*}$$

Conclusion:
Since we've shown $R_1 \leq R_2$ and $R_2 \leq R_1$, we can conclude that $R_1 = R_2$, which proves the equivalence of the two forms of the operator.
```

## Algorithms for Solving the Optimality Equations

The optimality equations are operator equations. Therefore, we can apply general numerical methods to solve them. Applying the successive approximation method to the Bellman optimality equation yields a method known as "value iteration" in dynamic programming. A direct application of the blueprint for successive approximation yields the following algorithm:

````{prf:algorithm} Value Iteration
:label: value-iteration

**Input** Given an MDP $(S, A, P, R, \gamma)$ and tolerance $\varepsilon > 0$  

**Output** Compute an $\varepsilon$-optimal value function $V$ and policy $\pi$  

1. Initialize $v_0(s) = 0$ for all $s \in S$  
2. $n \leftarrow 0$  
3. **repeat**  

    1. For each $s \in S$:  

        1. $v_{n+1}(s) \leftarrow \max_{a \in A} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)v_n(s')\right\}$  

    2. $\delta \leftarrow \|v_{n+1} - v_n\|_\infty$  
    3. $n \leftarrow n + 1$  

4. **until** $\delta < \frac{\varepsilon(1-\gamma)}{2\gamma}$  
5. For each $s \in S$:  

    1. $\pi(s) \leftarrow \arg\max_{a \in A} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)v_n(s')\right\}$  

6. **return** $v_n, \pi$  
````

The termination criterion in this algorithm is based on a specific bound that provides guarantees on the quality of the solution. This is in contrast to supervised learning, where we often use arbitrary termination criteria based on computational budget or early stopping when the learning curve flattens. This is because establishing implementable generalization bounds in supervised learning is challenging.

However, in the dynamic programming context, we can derive various bounds that can be implemented in practice. These bounds help us terminate our procedure with a guarantee on the precision of our value function and, correspondingly, on the optimality of the resulting policy.

````{prf:proposition} Convergence of Value Iteration 
:label: value-iteration-convergence
(Adapted from {cite:t}`Puterman1994` theorem 6.3.1)

Let $v_0$ be any initial value function, $\varepsilon > 0$ a desired accuracy, and let $\{v_n\}$ be the sequence of value functions generated by value iteration, i.e., $v_{n+1} = \mathrm{L}v_n$ for $n \geq 1$, where $\mathrm{L}$ is the Bellman optimality operator. Then:

1. $v_n$ converges to the optimal value function $v^*_\gamma$,
2. The algorithm terminates in finite time,
3. The resulting policy $\pi_\varepsilon$ is $\varepsilon$-optimal, and
4. When the algorithm terminates, $v_{n+1}$ is within $\varepsilon/2$ of $v^*_\gamma$.

````

````{prf:proof}
Parts 1. and 2. follow directly from the fact that $\mathrm{L}$ is a contraction mapping. Hence, by Banach's fixed-point theorem, it has a unique fixed point (which is $v^*_\gamma$), and repeated application of $\mathrm{L}$ will converge to this fixed point. Moreover, this convergence happens at a linear/geometric rate, which ensures that we reach the termination condition in finite time.

To show that the Bellman optimality operator $\mathrm{L}$ is a contraction mapping, we need to prove that for any two value functions $v$ and $u$:

$$\|\mathrm{L}v - \mathrm{L}u\|_\infty \leq \gamma \|V - U\|_\infty$$

where $\gamma \in [0,1)$ is the discount factor and $\|\cdot\|_\infty$ is the supremum norm.

Let's start by writing out the definition of $\mathrm{L}v$ and $\mathrm{L}u$:

   $$\begin{align*}(\mathrm{L}v)(s) &= \max_{a \in A} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)V(s')\right\}\\
   (\mathrm{L}u)(s) &= \max_{a \in A} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)U(s')\right\}\end{align*}$$

Now, for any state s, let $a_V$ be the action that achieves the maximum for $\mathrm{L}$V, and $a_U$ be the action that achieves the maximum for $\mathrm{L}u$. Then:

   $$\begin{align*}(\mathrm{L}v)(s)&= r(s,a_V) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a_V)V(s')\\
   (\mathrm{L}u)(s) &= r(s,a_U) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a_U)U(s')\end{align*}$$

By the definition of $a_V$ and $a_U$, we know that:

   $$\begin{align*}(\mathrm{L}v)(s) &\geq r(s,a_U) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a_U)V(s')\\
   (\mathrm{L}u)(s) &\geq r(s,a_V) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a_V)U(s')\end{align*}$$

Subtracting these inequalities:

   $$\begin{align*}(\mathrm{L}v)(s) - (\mathrm{L}u)(s) &\leq \gamma \sum_{j \in \mathcal{S}} p(j|s,a_V)(V(s') - U(s'))\\
   (\mathrm{L}u)(s) - (\mathrm{L}v)(s) &\leq \gamma \sum_{j \in \mathcal{S}} p(j|s,a_U)(U(s') - V(s'))\end{align*}$$

Taking the absolute value of both sides and using the fact that $\sum_{j \in \mathcal{S}} p(j|s,a) = 1$ for any s and a:

   $$|(\mathrm{L}v)(s) - (\mathrm{L}u)(s)| \leq \gamma \max_{j \in \mathcal{S}} |V(s') - U(s')| = \gamma \|V - U\|_\infty$$

Since this holds for all $s$, we can take the supremum over $s$:

   $$\|\mathrm{L}v - \mathrm{L}u\|_\infty \leq \gamma \|V - U\|_\infty$$

Thus, we have shown that $\mathrm{L}$ is indeed a contraction mapping with contraction factor $\gamma$.

Now, let's focus on parts 3. and 4. Suppose our algorithm has just terminated, i.e., $\|v_{n+1} - v_n\| < \frac{\varepsilon(1-\gamma)}{2\gamma}$ for some $n$. We want to show that our current value function $v_{n+1}$ and the policy $\pi_\varepsilon$ derived from it are close to optimal.

We start with the following inequality:

$$\|V^{\pi_\varepsilon}_\gamma - v^*_\gamma\| \leq \|V^{\pi_\varepsilon}_\gamma - v_{n+1}\| + \|v_{n+1} - v^*_\gamma\|$$

This inequality is derived using the triangle inequality:

$$\|V^{\pi_\varepsilon}_\gamma - v^*_\gamma\| = \|(V^{\pi_\varepsilon}_\gamma - v_{n+1}) + (v_{n+1} - v^*_\gamma)\| \leq \|V^{\pi_\varepsilon}_\gamma - v_{n+1}\| + \|v_{n+1} - v^*_\gamma\|$$

Let's focus on the first term, $\|V^{\pi_\varepsilon}_\gamma - v_{n+1}\|$. We can expand this:

$$
\begin{aligned}
\|V^{\pi_\varepsilon}_\gamma - v_{n+1}\| &= \|\mathrm{L}_{\pi_\varepsilon}V^{\pi_\varepsilon}_\gamma - v_{n+1}\| \\
&\leq \|\mathrm{L}_{\pi_\varepsilon}V^{\pi_\varepsilon}_\gamma - \mathrm{L}v_{n+1}\| + \|\mathrm{L}v_{n+1} - v_{n+1}\| \\
&= \|\mathrm{L}_{\pi_\varepsilon}V^{\pi_\varepsilon}_\gamma - \mathrm{L}_{\pi_\varepsilon}v_{n+1}\| + \|\mathrm{L}v_{n+1} - \mathrm{L}v_n\| \\
&\leq \gamma\|V^{\pi_\varepsilon}_\gamma - v_{n+1}\| + \gamma\|v_{n+1} - v_n\|
\end{aligned}
$$

Here, we've used the fact that $V^{\pi_\varepsilon}_\gamma$ is a fixed point of $\mathrm{L}_{\pi_\varepsilon}$, that $\mathrm{L}_{\pi_\varepsilon}v_{n+1} = \mathrm{L}v_{n+1}$ (by definition of $\pi_\varepsilon$), and that both $\mathrm{L}$ and $\mathrm{L}_{\pi_\varepsilon}$ are contractions with factor $\gamma$.

Rearranging this inequality, we get:

   $$\|V^{\pi_\varepsilon}_\gamma - v_{n+1}\| \leq \frac{\gamma}{1-\gamma}\|v_{n+1} - v_n\|$$

We can derive a similar bound for $\|v_{n+1} - v^*_\gamma\|$:

   $$\|v_{n+1} - v^*_\gamma\| \leq \frac{\gamma}{1-\gamma}\|v_{n+1} - v_n\|$$

Now, remember that our algorithm terminated when $\|v_{n+1} - v_n\| < \frac{\varepsilon(1-\gamma)}{2\gamma}$. Plugging this into our bounds:

   $$\|V^{\pi_\varepsilon}_\gamma - v_{n+1}\| \leq \frac{\gamma}{1-\gamma} \cdot \frac{\varepsilon(1-\gamma)}{2\gamma} = \frac{\varepsilon}{2}$$
   
   $$\|v_{n+1} - v^*_\gamma\| \leq \frac{\gamma}{1-\gamma} \cdot \frac{\varepsilon(1-\gamma)}{2\gamma} = \frac{\varepsilon}{2}$$

Finally, combining these results with our initial inequality:

   $$\|V^{\pi_\varepsilon}_\gamma - v^*_\gamma\| \leq \|V^{\pi_\varepsilon}_\gamma - v_{n+1}\| + \|v_{n+1} - v^*_\gamma\| \leq \frac{\varepsilon}{2} + \frac{\varepsilon}{2} = \varepsilon$$

This completes the proof. We've shown that when the algorithm terminates, our value function $v_{n+1}$ is within $\varepsilon/2$ of optimal (part 4.), and our policy $\pi_\varepsilon$ is $\varepsilon$-optimal (part 3.).
````
### Newton-Kantorovich applied to the Optimality Equations

Another perspective on the optimality equations is that instead of looking for $v_\gamma^\star$ as the fixed-point of $\mathrm{L}$ in the fixed-point problem find $v$ such that $\mathrm{L} v = v$, we consider instead the related form in the nonlinear operator equation $\mathrm{L}v - v = 0$. Writing things in this way highlights the fact that we could also consider using Newton-Kantorovich iteration to solve this equation instead of the successive approximation method.

If we were to apply the blueprint of Newton-Kantorovich, we would get:

````{prf:algorithm} Newton-Kantorovich for Bellman Optimality Equation
:label: nk-bellman

**Input:** MDP $(S, A, P, R, \gamma)$, initial guess $v_0$, tolerance $\epsilon > 0$

**Output:** Approximate optimal value function $v^\star$

1. Initialize $n = 0$
2. **repeat**
      
      3. Compute the Fréchet derivative of $\mathrm{B}(v) = \mathrm{L}v - v$ at $v_n$:
         
         $$\mathrm{B}'(v_n) = \mathrm{L}' - \mathrm{I}$$
         where $\mathrm{I}$ is the identity operator

      4. Solve the linear equation for $h_n$:

         $$(\mathrm{L}' - \mathrm{I})h_n = v_n - \mathrm{L}v_n$$

      5. Update: $v_{n+1} = v_n - h_n$

      6. $n = n + 1$
7. **until** $\|\mathrm{L}v_n - v_n\| < \epsilon$
8. **return** $v_n$
````

The key difference in this application is the specific form of our operator $\mathrm{B}(v) = \mathrm{L}v - v$, where $\mathrm{L}$ is the Bellman optimality operator. The Fréchet derivative of this operator is $\mathrm{B}'(v) = \mathrm{L}' - \mathrm{I}$, where $\mathrm{L}'$ is the Fréchet derivative of the Bellman optimality operator and $\mathrm{I}$ is the identity operator.

Let's examine the Fréchet derivative of $\mathrm{L}$ more closely. The Bellman optimality operator $\mathrm{L}$ is defined as:

$$(\mathrm{L}v)(s) \equiv \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in S} p(j|s,a)v(j)\right\}$$

For each state $s$, let $a^*(s)$ be the action that achieves the maximum in $(\mathrm{L}v)(s)$. Then, for any function $h$, the Fréchet derivative of $\mathrm{L}$ **at** v has the following effect when applied to any function $h$:

$$(\mathrm{L}'(v)h)(s) \equiv \gamma \sum_{j \in S} p(j|s,a^*(s))h(j)$$

This means that the Fréchet derivative $\mathrm{L}'(v)$ is a linear operator that, when applied to a function $h$, gives a new function whose value at each state $s$ is a discounted expectation of $h$ over the next states, using the transition probabilities corresponding to the optimal action $a^*(s)$ at the current point $v$.

Now, let's look more closely at what happens in each iteration of this Newton-Kantorovich method:

1. In step 3, when we compute $\mathrm{L}'(v_n)$, we're essentially defining a policy based on the current value function estimate. This policy chooses the action $a^*(s)$ for each state $s$.

2. In step 4, we solve the linear equation:

   $$(\mathrm{L}' - \mathrm{I})h_n = v_n - \mathrm{L}v_n$$
   
   This can be rearranged to:

   $$\mathrm{L}'h_n = \mathrm{L}v_n$$
   
   Solving this equation is equivalent to evaluating the policy defined by $a^*(s)$.

3. The update in step 5 can be rewritten as:

   $$v_{n+1} = \mathrm{L}v_n$$
   
   This step improves our value function estimate based on the policy we just evaluated.

Interestingly, this sequence of steps - deriving a policy from the current value function, evaluating that policy, and then improving the value function - is a well-known algorithm in dynamic programming called policy iteration. In fact, policy iteration can be viewed as simply the application of the Newton-Kantorovich method to the operator $\mathrm{B}(v) = \mathrm{L}v - v$.

### Policy Iteration 

While we derived policy iteration-like steps from the Newton-Kantorovich method, it's worth examining policy iteration as a standalone algorithm, as it has been traditionally presented in the field of dynamic programming.

The policy iteration algorithm for discounted Markov decision problems is as follows:

````{prf:algorithm} Policy Iteration
:label: policy-iteration

**Input:** MDP $(S, A, P, R, \gamma)$
**Output:** Optimal policy $\pi^*$

1. Initialize: $n = 0$, select an arbitrary decision rule $d_0 \in D$
2. **repeat**
   3. (Policy evaluation) Obtain $\mathbf{v}^n$ by solving:
   
      $$(\mathbf{I}-\gamma \mathbf{P}_{d_n}) \mathbf{v} = \mathbf{r}_{d_n}$$

   4. (Policy improvement) Choose $d_{n+1}$ to satisfy:

       $$d_{n+1} \in \arg\max_{d \in D}\left\{\mathbf{r}_d+\gamma \mathbf{P}_d \mathbf{v}^n\right\}$$
       
       Set $d_{n+1} = d_n$ if possible.

   5. If $d_{n+1} = d_n$, **return** $d^* = d_n$

   6. $n = n + 1$
7. **until** convergence
````

As opposed to value iteration, this algorithm produces a sequence of both deterministic Markovian decision rules $\{d_n\}$ and value functions $\{\mathbf{v}^n\}$. We recognize in this algorithm the linearization step of the Newton-Kantorovich procedure, which takes place here in the policy evaluation step 3 where we solve the linear system $(\mathbf{I}-\gamma \mathbf{P}_{d_n}) \mathbf{v} = \mathbf{r}_{d_n}$. In practice, this linear sytem could be solved either using direct methods (eg. Gaussian elimination), using  simple iterative methods such as the successive approximation method for policy evaluation, or more sophisticated methods such as GMRES. 
