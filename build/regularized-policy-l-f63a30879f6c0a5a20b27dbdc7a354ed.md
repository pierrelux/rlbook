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
# Regularized and Residual-Based Policy Learning

DDPG and TD3 address continuous actions by learning a deterministic policy that amortizes the $\arg\max_a q(s,a)$ operation. But deterministic policies have a fundamental limitation: they require external exploration noise (Gaussian perturbations in TD3) and can converge to suboptimal deterministic behaviors without adequate coverage of the state-action space.

Gradient estimators make stochastic actors trainable, while regularized dynamic
programming specifies the stochastic policy they should approach. Can these
pieces avoid both the continuous-action integral in the soft Bellman equation
and the repeated optimization of a hard maximum?

The [smoothing chapter](regularized-dp.md) presents an alternative: entropy-regularized MDPs, where the agent maximizes expected return plus a bonus for policy randomness. This yields stochastic policies with exploration built into the objective itself. The smooth Bellman operator replaces the hard max with a soft-max:

$$
v^*(s) = \frac{1}{\beta} \log \sum_{a \in \mathcal{A}} \exp\left(\beta \cdot q^*(s,a)\right)
$$

where $\beta = 1/\alpha$ is the inverse temperature and $\alpha$ is the entropy regularization weight. For finite action spaces, this log-sum-exp is easy to compute. But for continuous actions $\mathcal{A} \subset \mathbb{R}^m$, the sum becomes an integral:

$$
v^*(s) = \frac{1}{\beta} \log \int_{\mathcal{A}} \exp\left(\beta \cdot q^*(s,a)\right) da
$$

This integral is intractable. We face an infinite-dimensional sum over the continuous action space. The very smoothness that gives us stochastic policies creates a new computational barrier, distinct from but analogous to the $\arg\max$ problem in standard FQI.

## From Intractable Integral to Tractable Expectation

How can the continuous-action log-partition integral be rewritten as an
expectation under the policy being learned?

Soft actor-critic (SAC) {cite:p}`haarnoja2018soft,haarnoja2018sacapplications` exploits an equivalence between the intractable integral and an expectation. The optimal policy under entropy regularization is the Boltzmann distribution $\pi^*(a|s) \propto \exp(\beta \cdot q^*(s,a))$. Under this policy, the soft value function becomes:

$$
v^*(s) = \mathbb{E}_{a \sim \pi^*(\cdot|s)}\left[q^*(s,a) - \alpha \log \pi^*(a|s)\right]
$$

This converts the intractable integral into an expectation we can estimate by sampling. SAC learns a parametric policy $\pi_{\boldsymbol{\phi}}$ that approximates the Boltzmann distribution, enabling fast action selection via a single forward pass. For bootstrap targets, SAC samples $\tilde{a}' \sim \pi_{\boldsymbol{\phi}}(\cdot|s')$ and computes:

$$
y = r + \gamma \left[\min_{j=1,2} q^j_{\boldsymbol{\theta}_{\text{target}}}(s', \tilde{a}') - \alpha \log \pi_{\boldsymbol{\phi}}(\tilde{a}'|s')\right]
$$

The minimum over twin Q-networks applies the clipped double-Q trick from TD3. Exploration comes from the policy's stochasticity rather than external noise.

## Learning the Policy: Matching the Boltzmann Distribution

Which divergence or equivalent stochastic objective moves the actor toward the
Boltzmann distribution induced by the soft Q-function?

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

Can a soft-optimal policy be trained by enforcing multi-step consistency
directly rather than constructing one-step Bellman targets?

DDPG, TD3, and SAC all follow the same solution template from fitted Q-iteration: compute Bellman targets using the current Q-function, fit the Q-function to those targets, repeat. This is **successive approximation**, the function iteration approach $v_{k+1} = \Proj \Bellman v_k$ from the [projection methods chapter](approximate-bellman-equations.md).

Path Consistency Learning (PCL) {cite:p}`Nachum2017` solves the Bellman equation differently. Instead of iterating the operator, it directly minimizes a **residual**. This is the least-squares approach from projection methods: solve $\Residual(v) = 0$ by minimizing $\|\Residual(v)\|^2$. The method exploits special structure (smooth Bellman operators under deterministic dynamics) that conventional methods cannot leverage.

### The Path Consistency Property

Consider the entropy-regularized Q-function Bellman equation from the [smoothing chapter](regularized-dp.md). Under general stochastic dynamics, it involves an expectation over next states:

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

This is the least-squares residual approach from the [projection methods chapter](approximate-bellman-equations.md). SAC computes targets $y_i$ and fits to them (successive approximation). PCL directly minimizes the residual without computing targets or performing a separate fitting step.

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

{prf:ref}`pcl` uses separate networks for policy and value. But we can use a single Q-network $q_{\boldsymbol{\theta}}(s,a)$ and derive both:

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

## Model Predictive Path Integral Control

What does the same exponential weighting look like when action sequences are
sampled and optimized online instead of amortized into a network?

SAC and PCL both learn policies that approximate the Boltzmann distribution $\pi^*(a|s) \propto \exp(\beta \cdot q^*(s,a))$ induced by entropy regularization. This amortization allows fast action selection at deployment: a single forward pass through the policy network. An alternative approach forgoes learning entirely and instead performs optimization at every decision.

Model Predictive Path Integral control (MPPI) {cite:p}`williams2017mppi` uses the Boltzmann weighting directly for action sequence selection. Given a dynamics model $s_{t+1} = f(s_t, a_t)$ and current state $s_0$, MPPI samples $K$ action sequences $\{\boldsymbol{a}^{(i)}\}_{i=1}^K$, rolls them out to get costs $C^{(i)} = \sum_{t=0}^{H-1} c(s_t^{(i)}, a_t^{(i)})$, and computes the optimal action as a weighted average:

$$
a_0^* = \sum_{i=1}^K w^{(i)} a_0^{(i)}, \quad w^{(i)} = \frac{\exp(-C^{(i)}/\lambda)}{\sum_j \exp(-C^{(j)}/\lambda)}
$$

where $\lambda > 0$ is a temperature parameter. The weighting $w^{(i)} \propto \exp(-C^{(i)}/\lambda)$ is exactly the Boltzmann distribution. MPPI solves the entropy-regularized objective:

$$
\min_{\boldsymbol{a}} \mathbb{E}_{\boldsymbol{\xi}}\left[\sum_{t=0}^{H-1} c(s_t, a_t) + \lambda H(\pi)\right]
$$

where $\pi$ is the distribution over action sequences and $H(\pi) = -\mathbb{E}[\log \pi(\boldsymbol{a})]$ is entropy. The importance sampling estimate approximates the optimal action under this objective. The temperature $\lambda$ controls the trade-off between exploitation (focus on low-cost sequences) and exploration (maintain entropy).

```{prf:algorithm} Model Predictive Path Integral Control (MPPI)
:label: mppi

**Input:** Dynamics model $s_{t+1} = f(s_t, a_t)$, cost function $c(s,a)$, horizon $H$, number of samples $K$, temperature $\lambda$, noise distribution $\epsilon \sim \mathcal{N}(0, \Sigma)$

**Output:** Action $a_0^*$

1. Observe current state $s_0$
2. **for** $i = 1, \ldots, K$ **do**
    1. Sample action sequence: $a_t^{(i)} \leftarrow \bar{a}_t + \epsilon_t^{(i)}$ for $t = 0, \ldots, H-1$ $\quad$ // Perturb nominal
    2. Roll out: $s_{t+1}^{(i)} \leftarrow f(s_t^{(i)}, a_t^{(i)})$ for $t = 0, \ldots, H-1$
    3. Compute cost: $C^{(i)} \leftarrow \sum_{t=0}^{H-1} c(s_t^{(i)}, a_t^{(i)})$
3. Compute Boltzmann weights: $w^{(i)} \leftarrow \exp(-C^{(i)}/\lambda) / \sum_j \exp(-C^{(j)}/\lambda)$
4. **return** $a_0^* = \sum_{i=1}^K w^{(i)} a_0^{(i)}$
```

The algorithm samples perturbed action sequences around a nominal trajectory $\{\bar{a}_t\}$ (often the previous optimal sequence, shifted forward). The Boltzmann weights assign high probability to low-cost sequences. After executing $a_0^*$, the agent observes the next state and replans.

### MPPI as Non-Amortized Optimization

The contrast between MPPI and the methods in this chapter illuminates what amortization provides. SAC learns a policy $\pi_{\boldsymbol{\phi}}$ that approximates the Boltzmann distribution over actions at each state. PCL learns a Q-function from which the Boltzmann policy can be derived. Both invest computational effort during training to enable fast action selection at deployment: a single forward pass.

MPPI performs full optimization at every decision. At each state, it samples action sequences, weights them by exponentiated costs, and returns the weighted average. No learning occurs. The policy is implicitly defined by the optimization procedure itself.

This trade-off has practical consequences:

| **Aspect** | **Amortized (SAC, PCL)** | **Non-Amortized (MPPI)** |
|:-----------|:-------------------------|:-------------------------|
| **Action selection** | Single forward pass | $O(KH)$ model evaluations |
| **Generalization** | Policy generalizes across states | Optimization from scratch at each state |
| **Model requirement** | None (SAC) or deterministic (PCL) | Accurate dynamics model |
| **Approximation error** | Policy network approximation | None (exact optimization) |
| **Adaptability** | Requires retraining for new tasks | Adapts immediately to new cost functions |

MPPI excels at real-time control for systems with fast, accurate models (robotics, autonomous vehicles). The replanning handles model errors and disturbances without retraining. However, the per-step computation ($K \approx 100$-$1000$ rollouts) makes it expensive for complex dynamics or long horizons.

The entropy regularization that connects SAC, PCL, and MPPI is not coincidental. All three methods solve variants of the soft Bellman equation. SAC and PCL amortize the solution by learning value functions and policies. MPPI solves it directly through sampling. The Boltzmann weighting emerges in all cases as the optimal policy structure under entropy regularization.

## An Alternative: Euler Equation Methods

Can first-order optimality eliminate the value function and yield a functional
equation directly in the policy?

The methods developed in this chapter all parameterize policies, but they remain rooted in the Bellman equation. NFQCA, DDPG, TD3, and SAC learn Q-functions through successive approximation, then derive policies by maximizing these Q-functions. PCL minimizes a path residual derived from the soft Bellman equation. The policy serves as an amortized optimizer for a value-based objective.

There is a different approach, developed in computational economics {cite:p}`Judd1992,Rust1996,ndp`, that also parameterizes policies but solves an entirely different functional equation. Consider a control problem with continuous states and actions, deterministic dynamics $s' = f(s,a)$, and differentiable reward $r(s,a)$. The optimal action $\pi^*(s)$ satisfies the first-order condition:

$$
\frac{\partial r(s,a)}{\partial a}\Big|_{a=\pi^*(s)}
+
\gamma\, \frac{\partial v^*(s')}{\partial s'}\Big|_{s'=f(s,\pi^*(s))}\,
\frac{\partial f(s,a)}{\partial a}\Big|_{a=\pi^*(s)}
=
0.
$$

This Euler equation expresses optimality through derivatives rather than through the max operator. For problems with special structure (the Euler class, where dynamics are affine in the controlled state), envelope theorems eliminate $v^*$ entirely, yielding a closed functional equation $\mathcal{E}(\pi)(s) = 0$ in the policy alone.

With a parameterized policy $\pi_{\boldsymbol{\theta}}(s)$, we can discretize via collocation or Galerkin projection:

$$
G(\boldsymbol{\theta}) := \begin{bmatrix}
\mathcal{E}(\pi_{\boldsymbol{\theta}})(s_1) \\
\vdots \\
\mathcal{E}(\pi_{\boldsymbol{\theta}})(s_N)
\end{bmatrix} = 0.
$$

This is root-finding, not fixed-point iteration. Newton-type methods replace the successive approximation of fitted Q-iteration. The Euler operator is not a contraction, so convergence guarantees are problem-dependent.

What does this mean for reinforcement learning? The Euler approach shares the amortization idea: learn a policy network that directly outputs actions. But the training objective comes from first-order optimality conditions rather than from Bellman residuals or Q-function maximization. This raises questions worth considering. Could Euler-style objectives provide useful training signals for actor-critic methods? When dynamics are known or learned, could first-order conditions offer advantages over value-based objectives? The connection between these traditions remains underexplored.

## Summary

SAC converts the soft value integral into an expectation under a learned
stochastic policy. PCL instead minimizes a multi-step consistency residual, and
MPPI performs the corresponding Boltzmann-weighted search anew at every
decision. The Euler approach supplies a further alternative: solve a
first-order policy equation rather than a Bellman fixed point.

MPPI forgoes learning entirely, performing Boltzmann-weighted optimization at every decision. This avoids policy approximation error but requires $O(KH)$ model rollouts per action. SAC, PCL, and MPPI all solve entropy-regularized objectives; SAC and PCL amortize the solution while MPPI computes it directly.

All of these objectives retain an optimality equation derived from a value or
path-consistency relation. Can expected return be differentiated directly
without first treating a Bellman equation as the training objective? [Policy
gradients and actor-critic methods](policy-gradients.md) use the trajectory
score and bring value functions back as variance-reducing critics.

## Self-checks

:::{exercise} Compute trade-off
:label: ex-amortization-check-2

Contrast MPPI and an actor network in terms of online computation and approximation error.
:::

:::{solution} ex-amortization-check-2
:class: dropdown

MPPI spends many model rollouts at every decision and avoids a persistent actor approximation. An actor is cheap online but can introduce error because it only approximates the optimizer learned during training.
:::
