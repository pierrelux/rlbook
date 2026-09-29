# Model Predictive Path Integral Control

The preceding chapters formulated deterministic trajectory optimization as
a nonlinear program. Shooting and collocation turn the dynamics, costs, and
constraints into a finite-dimensional problem that we can solve with an NLP
solver. Gradient-based methods are effective when accurate derivatives are
available. How can we improve a trajectory when we can simulate the system
and evaluate its cost, but differentiating through that computation is
difficult or expensive?

Model predictive path integral control (MPPI) uses sampling to construct an
update. It draws candidate control sequences, simulates their trajectories,
and forms a weighted average that favors lower-cost outcomes. This requires
no derivatives of the dynamics or cost, and the candidate rollouts can run
in parallel. The derivation begins with a distribution over control
sequences: costs determine which sequences receive more probability, and
fitting a Gaussian to that distribution gives the weighted-average update.
This construction applies even to a deterministic simulator. An aircraft
example extends the calculation to uncertain winds by estimating each
candidate flight's expected cost before assigning its weight.
Executing the first action and planning again from the next observation
connects these computations to the following chapter on receding-horizon
control.

## Learning Goals

After working through the examples and exercises, you should be able to:

- Calculate an MPPI update, including the correction for sampling around a
  nonzero nominal control sequence.
- Derive exponential weighting from a KL-regularized objective and obtain
  the control update by fitting a Gaussian mean.
- Distinguish samples used to explore controls from samples used to
  estimate a candidate's expected cost under physical disturbances.
- Assess a sampled plan using feasibility checks, weight concentration,
  and comparisons under shared disturbances.
- Execute the first action of a sampled plan and update the remaining plan
  from new observations.

## Prerequisites

The algorithm uses the [single-shooting rollout](numerical-trajectory-optimization.md)
and the [finite-horizon optimal-control formulation](discrete-time-optimal-control.md).
The sampling derivation requires Gaussian distributions and expectations;
KL divergence and importance sampling are developed here. The aircraft
example uses conditional disturbance models from
[Stochastic Dynamics and Partial Observation](stochastic-dynamics-observation.md).

(sec-mppi)=
## Improving a Control Sequence by Sampling

We already know how to evaluate a control sequence: simulate the system
and compute the resulting cost. How can we improve that sequence? Instead
of computing a gradient, we try many random variations, evaluate each one,
and form a weighted average that gives more influence to the lower-cost
candidates. That average becomes our next control sequence.

Write the current sequence as $\mathbf U=(\mathbf u_0,\ldots,\mathbf u_{N-1})$.
Draw $K$ Gaussian perturbation sequences $\boldsymbol\epsilon^{(i)}$ and
simulate each candidate $\mathbf V^{(i)}=\mathbf U+\boldsymbol\epsilon^{(i)}$.
After assigning normalized weights $w_i$ to the resulting trajectories,
form the new control sequence

$$
\mathbf U^+=\sum_{i=1}^K w_i\mathbf V^{(i)}
=\mathbf U+\sum_{i=1}^K w_i\boldsymbol\epsilon^{(i)},
\qquad w_i\geq0,\qquad \sum_{i=1}^K w_i=1.
$$

The same trajectory weight multiplies every action in its sequence. How
should those weights depend on the costs? Giving every candidate the same
weight, $w_i=1/K$, would ignore the rollout results. Because the perturbations
have mean zero, this update would return $\mathbf U$ on average, regardless
of which candidates performed well. At the other extreme, assigning weight
one to the lowest-cost candidate and zero to the others would select a
single sequence. A candidate with a nearly identical cost would have no
influence, and a small change in which candidate ranks first could switch
the entire update.

MPPI allows several low-cost candidates to contribute, with their influence
decreasing smoothly as cost increases. Consider first the initial sequence
$\mathbf U=\mathbf0$. The candidate sequences are then the zero-mean
Gaussian perturbations themselves. This Gaussian is the **reference
distribution**: it remains fixed even when later updates move the sampling
mean away from zero. Let $S_i$ be the accumulated state cost along rollout
$i$, including its terminal cost. MPPI assigns it the positive score $\exp(-S_i/\lambda)$
and divides by the sum of all scores:

$$
w_i=\frac{\exp(-S_i/\lambda)}{\sum_{j=1}^K\exp(-S_j/\lambda)},
\qquad \lambda>0.
$$

This is the softmax of the negative costs, with temperature $\lambda$.
The minus sign makes lower costs receive larger scores, and the denominator
makes the weights sum to one. The parameter $\lambda$ has the same units as
cost, so $S_i/\lambda$ is dimensionless. To see how it sets the strength of
the preference, compare two candidates. Their common denominator cancels:

$$
\frac{w_i}{w_j}
=\frac{\exp(-S_i/\lambda)}{\exp(-S_j/\lambda)}
=\exp\!\left(-\frac{S_i-S_j}{\lambda}\right).
$$

If rollout $i$ costs $\lambda\log2$ more than rollout $j$, it receives half
as much weight. Thus cost differences, rather than the arbitrary zero of
the cost scale, determine relative influence. For a fixed batch, small
$\lambda$ makes even modest cost differences strongly affect the weights;
large $\lambda$ makes them less consequential. As $\lambda\to0^+$, the
weights concentrate on the lowest-cost candidate, or split equally among
tied minima. As $\lambda\to\infty$, they approach $1/K$. These are the two
alternatives above, now connected by a continuous weighting rule.

The curves in @fig-mppi-exponential-weights show this dependence on cost
and temperature. Each curve compares a rollout's weight with the weight of
the best candidate in the same batch.

:::{figure} _static/brownian_mppi/exponential-weights.svg
:label: fig-mppi-exponential-weights
:alt: Three exponential curves plot a rollout's weight relative to the best rollout against its excess cost. The curve for temperature one half falls fastest, temperature one falls at an intermediate rate, and temperature two falls slowest. A horizontal dotted line shows the equal-weight limit. At temperature one, an excess cost of log two gives half the best rollout's weight.

Smaller temperatures reduce the influence of higher-cost trajectories more
sharply. With $S_{\min}=\min_i S_i$, the horizontal axis is the excess cost
$\Delta S=S_i-S_{\min}$, and the vertical axis is the ratio
$w_i/w_{\mathrm{best}}=\exp(-\Delta S/\lambda)$. Costs and temperatures use
the same arbitrary cost unit. The marked point shows that $\Delta S=\log2$
halves the relative weight when $\lambda=1$. The dotted horizontal line is
the equal-weight limit as $\lambda\to\infty$. A ratio of one at
$\Delta S=0$ does not mean that the best rollout receives all the normalized
weight: the actual $w_i$ still sum to one across the batch.
:::

The Gaussian reference already favors small controls. Its contribution to
control effort will appear explicitly in the distributional objective
below; $S_i$ includes only state and terminal costs. The displayed weights
apply when sampling from this reference, with $\mathbf U=\mathbf0$.
Sampling around a nonzero plan changes how often each candidate is drawn
and requires a corresponding correction to its weight.

The computational attraction is that each rollout and cost evaluation can
run independently, making large batches suitable for GPU execution
{cite:p}`Williams2017InformationTheoretic`. This parallelism does not
guarantee that a modest number of samples will find low-cost control sequences: long
horizons, many control variables, and narrow feasible regions can make the
search difficult. We will check both sampling quality and feasibility when
using the update in the aircraft example.

So far, exponential weighting is a rule for combining a batch of trials.
To justify the rule, we need an objective whose solution gives those
weights, and a reason to use their average as the next plan. Both follow
by treating the candidate sequences as samples from a distribution and
optimizing that distribution before fitting a new control sequence to it.

(sec-mppi-distribution)=
## From Trajectory Weights to MPPI

Which optimization problem gives the exponential weights and their
weighted-average update? A distribution over control sequences lets us
balance low trajectory cost against a preference for small inputs. The
resulting distribution need not have a simple form, but fitting a Gaussian
to it produces a mean that can serve as the next plan. Sampling then
estimates this mean. Throughout this derivation, randomness comes from
the search over controls; the simulator itself can be deterministic.

### Reference and sampling distributions

For a general rollout, write
$\mathbf{x}_{k+1}=\mathbf{f}_k(\mathbf{x}_k,\mathbf{v}_k)$ and collect the
candidate inputs into
$\mathbf{V}=(\mathbf{v}_0,\ldots,\mathbf{v}_{N-1})$. The initial state is
fixed, so each input sequence determines a simulated state sequence.
Define the state and terminal cost of a rollout by

$$
S(\mathbf V)=c_N(\mathbf{x}_N)
+\sum_{k=0}^{N-1}c_k(\mathbf{x}_k).
$$

Each $c_k$ is the cost for one interval; if it comes from a continuous
running cost, its discretization includes the interval length. Control
effort will enter through the distribution over inputs.

Choose a Gaussian reference distribution $P_0$, centered at zero, and a
Gaussian sampling distribution $Q_{\mathbf U}$, centered at the current
nominal sequence. Both have the same positive-definite covariance $\Sigma$:

$$
p_0(\mathbf V)=\prod_{k=0}^{N-1}\mathcal N(\mathbf v_k;\mathbf0,\Sigma),
\qquad
q_{\mathbf U}(\mathbf V)=
\prod_{k=0}^{N-1}\mathcal N(\mathbf v_k;\mathbf u_k,\Sigma).
$$

The reference favors small inputs and remains fixed during a planning
update. The sampling mean $\mathbf U=(\mathbf u_0,\ldots,\mathbf u_{N-1})$
can move as the current plan improves. For the initial update the two
distributions coincide, since $\mathbf U=0$. Later, samples can concentrate
near a nonzero plan while the reference continues to define the same
objective. Keeping these roles separate allows the sampler to improve
without changing the problem it is trying to solve.

### Exponentially weighting the reference

The reference describes input preferences before the state costs are
considered. Multiplying it by the exponential cost weight favors trajectories
with lower state and terminal costs. The batch weights extend to a density
over all input sequences:

$$
p^\star(\mathbf V)=\frac{e^{-S(\mathbf V)/\lambda}p_0(\mathbf V)}{Z},
\qquad
Z=\mathbb E_{P_0}[e^{-S(\mathbf V)/\lambda}],
\qquad \lambda>0.
$$

Assume $S$ is measurable and bounded below and $0<Z<\infty$.
The normalized result is a **Gibbs distribution**: two trajectories with
equal reference density and cost difference $\Delta S$ have density ratio
$e^{-\Delta S/\lambda}$. The parameter $\lambda$ has the units of cost
and controls how strongly those costs affect the ratio.

The factor $p_0$ matters even though it did not appear in the opening
batch weights: sampling from $P_0$ already makes high-density regions
appear more often in the batch. Each of those draws then receives the
additional factor $e^{-S/\lambda}$. The density $p^\star$ describes the
combined effect of sampling frequency and cost weighting.

### The objective defined by the weights

The exponential weights solve an optimization problem over distributions
of input sequences. We seek a distribution $Q$ that reduces expected
state and terminal cost while paying a penalty for departing from $P_0$.
Measure that departure with the **Kullback--Leibler divergence**. For
densities $q,p$, it is

$$
D_{\mathrm{KL}}(Q\|P)
=\mathbb E_{\mathbf V\sim Q}\left[\log\frac{q(\mathbf V)}{p(\mathbf V)}\right].
$$

This divergence is nonnegative and equals zero only when the two
distributions agree. For it to be finite, $Q$ must put no probability where
$P$ has zero density, a condition called absolute continuity. With
$\lambda>0$ setting the cost of departure from the reference, the first
optimization problem is

```{math}
:label: eq-mppi-distribution-objective
\min_Q\;\mathcal J(Q),
\qquad
\mathcal J(Q)
=\underbrace{\mathbb E_{\mathbf V\sim Q}[S(\mathbf V)]}_{\text{expected state and terminal cost}}
+\underbrace{\lambda D_{\mathrm{KL}}(Q\|P_0)}_{\text{penalty for departing from the reference}}.
```

Here $Q$ ranges over distributions on complete input sequences for which
both terms are finite; it is not restricted to a Gaussian family.

To verify that $P^\star$ solves this problem, we can express the difference
between $\mathcal J(Q)$ and its minimum as a nonnegative divergence. The
definition $p^\star=e^{-S/\lambda}p_0/Z$ gives

$$
\log\frac{q(\mathbf V)}{p^\star(\mathbf V)}
=\log\frac{q(\mathbf V)}{p_0(\mathbf V)}
+\frac{S(\mathbf V)}{\lambda}+\log Z.
$$

Taking the expectation under $Q$ recovers the two terms in
$\mathcal J(Q)$:

$$
\begin{aligned}
D_{\mathrm{KL}}(Q\|P^\star)
&=D_{\mathrm{KL}}(Q\|P_0)
+\frac{1}{\lambda}\mathbb E_Q[S(\mathbf V)]+\log Z\\
&=\frac{\mathcal J(Q)}{\lambda}+\log Z.
\end{aligned}
$$

Rearranging isolates the objective:

$$
\mathcal J(Q)=-\lambda\log Z
+\lambda D_{\mathrm{KL}}(Q\|P^\star)
\geq-\lambda\log Z.
$$

The quantity $Z$ is fixed by $S$, $P_0$, and $\lambda$, so the first term
does not depend on $Q$. The second term is zero at $Q=P^\star$ and positive
for any other distribution. Consequently,
$P^\star=\operatorname*{arg\,min}_Q\mathcal J(Q)$, with minimum value
$-\lambda\log Z$. The exponential weighting therefore gives the
distribution that minimizes expected state and terminal cost plus the KL
penalty in @eq-mppi-distribution-objective.

For a Gaussian $Q_{\mathbf U}$ with the same covariance as the reference,
the KL penalty is a quadratic function of its mean controls:

$$
\lambda D_{\mathrm{KL}}(Q_{\mathbf U}\|P_0)
=\frac\lambda2\sum_{k=0}^{N-1}
\mathbf u_k^\top\Sigma^{-1}\mathbf u_k.
$$

For example, choose $\lambda\Sigma^{-1}=hR$, where $h$ is the interval
length and $R\succ0$ specifies how costly each input direction is. Then
this penalty is $\tfrac12 h\sum_k\mathbf u_k^\top R\mathbf u_k$, the
discretization of the control-effort integral
$\tfrac12\int\mathbf u(t)^\top R\mathbf u(t)\,dt$ for controls held
constant on each interval. Thus, within this Gaussian family, departing
from the zero-mean reference incurs a quadratic cost on the nominal
controls. This accounts for control effort without adding it to $S$.
Changing $\lambda$ with $\Sigma$ fixed changes this penalty as well as
the concentration of the weights.

The solution $P^\star$ assigns probabilities to complete input sequences.
A controller still needs a nominal sequence to store and execute. We obtain
that sequence by approximating $P^\star$ with a Gaussian whose mean can be
stored as a plan.

### Fitting a Gaussian mean

The density $p^\star(\mathbf V)=e^{-S(\mathbf V)/\lambda}p_0(\mathbf V)/Z$
is generally neither Gaussian nor independent across time. To approximate
this distribution, consider the same Gaussian family as our sampler, with
a candidate mean sequence $\mathbf M=(\mathbf m_0,\ldots,\mathbf m_{N-1})$:

$$
q_{\mathbf M}(\mathbf V)
=\prod_{k=0}^{N-1}\mathcal N(\mathbf v_k;\mathbf m_k,\Sigma).
$$

Only the mean is adjustable; the covariance $\Sigma$ remains fixed.
The second optimization problem chooses the Gaussian that minimizes its
divergence from $P^\star$ in the following direction:

```{math}
:label: eq-mppi-gaussian-objective
\mathbf U^+
=\operatorname*{arg\,min}_{\mathbf M}
D_{\mathrm{KL}}(P^\star\|Q_{\mathbf M}).
```

Here $P^\star$ is fixed, and the variable is a finite-dimensional mean
sequence $\mathbf M$. Expanding this objective separates the two log
densities:

$$
D_{\mathrm{KL}}(P^\star\|Q_{\mathbf M})
=\mathbb E_{P^\star}[\log p^\star(\mathbf V)]
-\mathbb E_{P^\star}[\log q_{\mathbf M}(\mathbf V)].
$$

The first expectation does not depend on $\mathbf M$. In the second,
the Gaussian normalization also stays fixed because $\Sigma$ is fixed.
Substituting the Gaussian log density therefore gives

$$
D_{\mathrm{KL}}(P^\star\|Q_{\mathbf M})
=C+\frac12\mathbb E_{P^\star}
\left[\sum_{k=0}^{N-1}
(\mathbf v_k-\mathbf m_k)^\top\Sigma^{-1}
(\mathbf v_k-\mathbf m_k)\right].
$$

The constant $C$ collects the terms in the divergence that do not depend
on $\mathbf M$. Minimizing the divergence is thus equivalent to minimizing
the expected squared distance from the candidate mean to an input sequence
drawn from $P^\star$, with distances measured by $\Sigma^{-1}$. Assuming
finite second moments, differentiation gives

$$
\nabla_{\mathbf m_k}D_{\mathrm{KL}}(P^\star\|Q_{\mathbf M})
=\Sigma^{-1}\left(\mathbf m_k-\mathbb E_{P^\star}[\mathbf v_k]\right).
$$

Setting this gradient to zero yields the unique minimizer, since
$\Sigma^{-1}$ is positive definite:

$$
\mathbf u_k^+=\mathbb E_{P^\star}[\mathbf v_k],
\qquad k=0,\ldots,N-1.
$$

The updated Gaussian has the same mean as $P^\star$, the distribution
obtained by multiplying the reference density by the exponential cost
weight and normalizing. Its covariance remains $\Sigma$. It can therefore
lose features of $P^\star$: if low-cost sequences form separate groups,
their average can lie between those groups.

The first problem, @eq-mppi-distribution-objective, chooses an entire
distribution by minimizing $\mathbb E_Q[S(\mathbf V)]+\lambda
D_{\mathrm{KL}}(Q\|P_0)$. The second, @eq-mppi-gaussian-objective, holds
its solution $P^\star$ fixed and chooses a Gaussian mean by minimizing
$D_{\mathrm{KL}}(P^\star\|Q_{\mathbf M})$. If we instead restricted the
first problem to Gaussians, the identity above would give

$$
\operatorname*{arg\,min}_{\mathbf M}
\left\{\mathbb E_{Q_{\mathbf M}}[S(\mathbf V)]
+\lambda D_{\mathrm{KL}}(Q_{\mathbf M}\|P_0)\right\}
=\operatorname*{arg\,min}_{\mathbf M}
D_{\mathrm{KL}}(Q_{\mathbf M}\|P^\star).
$$

The order of the distributions in this divergence is reversed relative
to @eq-mppi-gaussian-objective. KL divergence is not symmetric, so these
two choices of Gaussian mean need not agree
{cite:p}`Williams2017InformationTheoretic`.

### A one-step quadratic check

For a scalar system $x_1=x_0+v$ with $x_0=0$ and target $1$, take
$S(v)=\tfrac\kappa2(v-1)^2$ with $\kappa>0$, $P_0=\mathcal N(0,1)$,
and $\lambda=1$.
The dynamics are deterministic: different endpoints result from trying
different inputs. Multiplying the reference density by $e^{-S(v)}$ and
completing the square gives

$$
p^\star(v)\propto
\exp\!\left[-\frac12\bigl(v^2+\kappa(v-1)^2\bigr)\right]
\propto\exp\!\left[-\frac{1+\kappa}{2}
\left(v-\frac\kappa{1+\kappa}\right)^2\right].
$$

Thus the first optimization problem has solution
$P^\star=\mathcal N(\kappa/(1+\kappa),1/(1+\kappa))$. Weighting has
shifted the reference toward the target and reduced its variance. The
second problem keeps the variance at its chosen value of one and adjusts
only the Gaussian mean:

$$
u^+=\operatorname*{arg\,min}_{m}
D_{\mathrm{KL}}\!\left(
\mathcal N\!\left(\frac{\kappa}{1+\kappa},\frac{1}{1+\kappa}\right)
\,\middle\|\,\mathcal N(m,1)\right)
=\frac{\kappa}{1+\kappa}.
$$

The resulting approximation is $Q_{u^+}=\mathcal N(\kappa/(1+\kappa),1)$.
It has the correct mean but is broader than $P^\star$.
Figure @fig-mppi-gaussian-mean shows both optimization steps for
$\kappa=2\log2$.

:::{figure} _static/brownian_mppi/gaussian-mean.svg
:label: fig-mppi-gaussian-mean
:alt: Two density plots share the same input and probability-density scales. On the left, multiplying a zero-mean Gaussian reference by the exponential cost weight produces a narrower Gaussian centered at about 0.581. On the right, the approximating Gaussian shares that mean but retains the reference variance of one, so it is broader than the cost-weighted distribution.

The two MPPI optimization problems change different properties of the
input distribution. For the scalar rollout $x_1=v$, target $1$, cost
$S(v)=\tfrac\kappa2(v-1)^2$, $\kappa=2\log2$, and $\lambda=1$, the
reference is $P_0=\mathcal N(0,1)$. Left: minimizing expected cost plus
$D_{\mathrm{KL}}(Q\|P_0)$ gives the blue solid density
$p^\star(v)\propto e^{-S(v)}p_0(v)$, with mean $u^+\approx0.581$ and
variance $1/(1+\kappa)\approx0.419$. The gray dashed curve is the reference.
Right: minimizing $D_{\mathrm{KL}}(P^\star\|Q_m)$ over $Q_m=\mathcal N(m,1)$
gives the orange dash-dotted density $q_{u^+}$, with the same mean as
$p^\star$ and variance one. The vertical dotted lines mark the means.
Both panels show analytic densities, without sampling error.
:::

In this quadratic example, $u^+$ also minimizes the deterministic
objective $\tfrac\kappa2(u-1)^2+\tfrac12u^2$: setting its derivative
$\kappa(u-1)+u$ to zero gives $u=\kappa/(1+\kappa)$.
This agreement checks the update in the example; taking the mean of
$P^\star$ need not minimize the deterministic cost for a general rollout.

To check the finite-sample calculation, set $\kappa=2\log2$ and start
from $u=0$. Suppose three draws from the reference, rounded for this
calculation, are $-1,0,1$. Evaluating their costs gives

| Candidate input $v$ | Endpoint $x_1$ | Cost $S(v)$ | Unnormalized weight | Normalized weight |
| --- | --- | --- | --- | --- |
| $-1$ | $-1$ | $4\log 2$ | $1/16$ | $1/25$ |
| $0$ | $0$ | $\log 2$ | $1/2$ | $8/25$ |
| $1$ | $1$ | $0$ | $1$ | $16/25$ |

The weighted average is

$$
u^+=\frac1{25}(-1)+\frac8{25}(0)+\frac{16}{25}(1)=\frac35.
$$

This three-sample estimate, $0.6$, is close to the exact mean
$2\log2/(1+2\log2)\approx0.581$. All three candidates contribute; the
update does not simply select the best sampled input, $1$. Sampling around
the updated plan will produce a different mix of candidates, so preserving
the same target mean requires adjusting their weights.

### Correcting for where the samples were drawn

Samples are available from $Q_{\mathbf U}$, rather than directly from
$P^\star$. Multiplying and dividing by the proposal density expresses the
desired mean as a ratio of expectations:

$$
\mathbf u_k^+
=\frac{\mathbb E_{Q_{\mathbf U}}
\left[e^{-S(\mathbf V)/\lambda}
\frac{p_0(\mathbf V)}{q_{\mathbf U}(\mathbf V)}\mathbf v_k\right]}
{\mathbb E_{Q_{\mathbf U}}
\left[e^{-S(\mathbf V)/\lambda}
\frac{p_0(\mathbf V)}{q_{\mathbf U}(\mathbf V)}\right]}.
$$

For example, moving a scalar proposal to a positive mean makes positive
inputs more common even if the target distribution has not changed. Weighting
only by state cost would mistake that extra sampling frequency for additional
target probability. The factor $p_0/q_{\mathbf U}$ removes it. Since both Gaussians have full support, every
trajectory with positive target density remains accessible. Writing
$\mathbf v_k=\mathbf u_k+\boldsymbol\epsilon_k$ and expanding their
quadratic exponents gives

$$
\begin{aligned}
\log\frac{p_0(\mathbf U+\boldsymbol\epsilon)}
{q_{\mathbf U}(\mathbf U+\boldsymbol\epsilon)}
&=-\frac12\sum_k\left[
(\mathbf u_k+\boldsymbol\epsilon_k)^\top\Sigma^{-1}
(\mathbf u_k+\boldsymbol\epsilon_k)
-\boldsymbol\epsilon_k^\top\Sigma^{-1}\boldsymbol\epsilon_k
\right]\\
&=-\sum_k\left[
\mathbf u_k^\top\Sigma^{-1}\boldsymbol\epsilon_k
+\frac12\mathbf u_k^\top\Sigma^{-1}\mathbf u_k\right].
\end{aligned}
$$

The last quadratic term is identical across samples and cancels from the
normalized weights. Consequently each sampled sequence $i$ receives the
corrected cost

$$
\widetilde S_i=S(\mathbf U+\boldsymbol\epsilon^{(i)})
+\lambda\sum_{k=0}^{N-1}
\mathbf u_k^\top\Sigma^{-1}\boldsymbol\epsilon_k^{(i)}.
$$

### A stable finite-sample update

For $K$ independent samples, subtract
$\rho=\min_i\widetilde S_i$ before exponentiation. This common subtraction
leaves normalized weights unchanged and avoids overflow:

$$
w_i=\frac{e^{-(\widetilde S_i-\rho)/\lambda}}
{\sum_{j=1}^K e^{-(\widetilde S_j-\rho)/\lambda}},
\qquad
\mathbf u_k^+=\mathbf u_k+
\sum_{i=1}^K w_i\boldsymbol\epsilon_k^{(i)}.
$$

The same full-trajectory weight updates every time slot. This ratio of
sample averages is a self-normalized importance-sampling estimator. It
is generally biased for finite $K$ and converges to the target mean when
the relevant weighted moments are integrable. Further iterations move
the proposal toward useful trajectories; with exact expectations, the
target mean would already be independent of the proposal.

A quadratic rollout makes the effect of this correction explicit.
Take $x_{k+1}=x_k+hv_k$, $x_0=0$, $N=10$, $h=0.1$, and terminal cost
$2(x_N-1)^2$. With $\lambda=0.5$ and $\Sigma=5$, the exact target mean
is $0.8$ at every time step. Sampling around $u_k=0.4$ and omitting the
correction instead estimates $0.88$, even with arbitrarily many samples.
Exercise @ex-mppi-likelihood-correction derives both limits. Moving the
proposal should change the accuracy of the estimate, not the mean being
estimated.

### Receding-horizon execution

The update gives a sequence of actions for a particular initial state.
Model error or physical disturbances can make the next observed state
differ from the rollout prediction. Continuing with the stored sequence
would ignore that discrepancy. Instead, apply only the first action and use
the newly observed state as the initial condition for another planning
problem. MPPI repeats the trajectory computation from that condition,
as in the sampling controllers of {cite:t}`williams2017mppi`.

```{prf:algorithm} Model Predictive Path Integral Control
:label: mppi

**Inputs:** Measured state $\mathbf x$, horizon $N$, rollout maps
$\mathbf f_k$, state costs $c_k$, terminal cost $c_N$, covariance
$\Sigma\succ0$, parameter $\lambda>0$, sample count $K$, and a finite
number $I$ of sampling iterations per observation.

**Initialization:** Set the nominal sequence $\mathbf U$ to zero or to an
available initial guess.

1. At each observation, repeat the following $I$ times:
   1. Save the current nominal sequence $\mathbf U$.
   2. Independently for $i=1,\ldots,K$:
      1. Draw $\boldsymbol\epsilon_k^{(i)}\sim\mathcal N(\mathbf0,\Sigma)$
         for $k=0,\ldots,N-1$.
      2. Set $\mathbf x_0^{(i)}=\mathbf x$ and $S_i=0$.
      3. For $k=0,\ldots,N-1$, accumulate
         $S_i\leftarrow S_i+c_k(\mathbf x_k^{(i)})$ and simulate
         $\mathbf x_{k+1}^{(i)}=
         \mathbf f_k(\mathbf x_k^{(i)},
         \mathbf u_k+\boldsymbol\epsilon_k^{(i)})$.
      4. Add the terminal cost and likelihood correction:
         $\widetilde S_i=S_i+c_N(\mathbf x_N^{(i)})
         +\lambda\sum_k\mathbf u_k^\top\Sigma^{-1}
         \boldsymbol\epsilon_k^{(i)}$.
   3. Compute $\rho=\min_i\widetilde S_i$ and normalize
      $w_i\propto\exp[-(\widetilde S_i-\rho)/\lambda]$.
   4. Update all controls simultaneously:
      $\mathbf u_k\leftarrow\mathbf u_k+
      \sum_iw_i\boldsymbol\epsilon_k^{(i)}$.
2. Apply $\mathbf u_0$ for one control interval and obtain the next state
   observation.
3. Shift the remaining sequence one slot forward, repeat its final control
   to fill the last slot, and return to step 1.
```

This version assumes unrestricted Gaussian inputs and finite rollout
costs. Nonconvex state constraints need additional treatment: even an
average of inputs from feasible trajectories can generate an infeasible
trajectory. Clipping samples also changes their density, so the displayed
Gaussian likelihood ratio does not apply unchanged to the clipped
variables. A chosen exploration covariance different from the reference
covariance requires the full corresponding density ratio.

The rollouts can be simulated in parallel. Their effectiveness still
depends on overlap between proposal and target. The quantity
$K_{\mathrm{eff}}=1/\sum_i w_i^2$ summarizes weight concentration: it is
$K$ for equal weights and approaches one when a single rollout dominates.
It diagnoses sampling concentration, rather than certifying optimality
or constraint satisfaction. A finite-sample update can increase the
rollout cost, particularly when only a few samples carry most of the
weight. Concentrated weights suggest increasing the sample count or
improving overlap between proposal and target. Increasing $\lambda$ with
a fixed reference weakens state-cost weighting, but also changes the
optimization objective. Altering the proposal covariance alone requires
the corresponding likelihood correction. Receding-horizon execution
supplies feedback through repeated observations; the stored sequence
itself is open loop.

### Connection to diffusion-based planning

The distribution over low-cost plans also connects this computation to
modern diffusion-based trajectory optimization. In model-based diffusion,
the objective defines a target density through an exponential cost weight.
Gaussian perturbations and cost-weighted averages then provide estimates
used to refine trajectories through a sequence of noise levels
{cite:p}`Pan2024ModelBasedDiffusion`. MPPI's weighted mean provides a useful
starting point for understanding that construction. The MPPI update here
uses a fixed sampling covariance and does not train a denoising network or
run a diffusion noise schedule. In both algorithms, random perturbations
can serve only to explore controls. Physical disturbances introduce a
separate source of variation: a single candidate plan can produce different
costs under different disturbance realizations. The aircraft example uses
several simulated winds to estimate the cost assigned to each candidate.

(sec-aircraft-mppi)=
## Aircraft Planning Under Uncertain Winds

Consider an Airbus A320 flying from Montréal–Trudeau (CYUL) to Toronto–Pearson
(CYYZ). The task is to choose climb, cruise, descent, and flight duration to
use little fuel while arriving within 1 km horizontally and 30 m vertically
of the destination. A favorable wind at one altitude may justify climbing,
but its later evolution is uncertain. How can the sampled-control update
account for these different possible outcomes when comparing candidate
flights?

The method from @sec-mppi-distribution can optimize an estimated expected
cost from a simulator. It requires separate samples for candidate controls
and for future winds: several winds must be evaluated before a candidate is
assigned its cost weight. The following model makes those two random
quantities explicit.

### The airborne model and its wind state

In the OpenAP A320 point-mass model {cite:p}`Sun2022`, the state is
$(p_E,p_N,h,m,t)$: east and north position in a local map,
altitude in meters, mass in kilograms, and elapsed time in seconds.
Here $h$ denotes altitude; numerical time steps will be given in seconds.
Controls are Mach number $M$, vertical speed $v_z$ in meters per second, and heading
$\chi$ in radians clockwise from north. If $v=\mathrm{TAS}(M,h)$ is true
airspeed and $\gamma=\arctan(v_z/v)$, the model is

$$
\begin{aligned}
\dot p_E &= v\cos\gamma\sin\chi+\bar w_E(p_E,p_N,h)+g_E,\\
\dot p_N &= v\cos\gamma\cos\chi+\bar w_N(p_E,p_N,h)+g_N,\\
\dot h &= v_z,\qquad
\dot m=-f_{\mathrm{fuel}}(m,v,h,v_z),\qquad \dot t=1.
\end{aligned}
$$

The horizontal equations separate air-relative motion from wind-driven
displacement: $\bar w$ is the fixed mean wind field and $g$ is a gust.
Fuel flow $f_{\mathrm{fuel}}$ reduces mass as the aircraft flies. The heading
convention and flight-path-angle approximation reproduce `OpenAP.top`'s
point-mass equations. Conversions to knots, feet, and feet per
minute occur only when calling OpenAP interfaces that require them. The
numerical derivatives are tested against `OpenAP.top`'s original symbolic
model at climb, cruise, and descent states. Optimization itself uses forward
simulation and Gaussian sampling.

The reference flight covers the complete airborne leg, with the following
initial and nominal conditions:

| Quantity | Value |
| --- | --- |
| Initial mass | $85\%$ of maximum takeoff mass |
| Departure and arrival altitude | $100$ ft |
| Nominal cruise altitude | $7000$ m |
| Nominal duration | $3600$ s |

Taxi, runway operations, and traffic are outside this model. Mach number,
vertical speed, heading, altitude, mass, and the geographic flight corridor have
explicit bounds. Thrust, lift, and excess-energy inequalities reproduce the
approximate `OpenAP.top` envelope checks. Adjacent command changes are also
checked against limits of $0.2$ Mach, $500$ ft/min in vertical speed, and
$15$ degrees in heading. These are constraints of the teaching model.

The mean field $\bar w$ is the committed ERA5 pressure-level snapshot for
June 1, 2023, at 12:00 UTC {cite:p}`ERA5PressureLevels2018`. Preprocessing converts the
original GRIB values once to a numerical array. The simulator interpolates
in pressure, latitude, and longitude, using ISA pressure at the aircraft's
altitude. Outside the snapshot's pressure range of $200$–$925$ hPa, it holds
the boundary value; in particular, winds near the ground use the lowest
available pressure level. The snapshot remains fixed in time during a flight.

A gust observed now should influence the near-future forecast, but it should
not determine the wind for the entire flight. An Ornstein–Uhlenbeck model
represents this persistence and gradual loss of predictability for
$g=(g_E,g_N)$. Its exact transition over a time increment $\Delta$ is

$$
g_{t+\Delta}=e^{-\Delta/\tau}g_t+
\sqrt{1-e^{-2\Delta/\tau}}\,
\begin{bmatrix}\sigma_E&0\\0&\sigma_N\end{bmatrix}\xi,
\qquad \xi\sim\mathcal N(0,I),
$$

with $\tau=300$ s, $\sigma_E=5$ m/s, and $\sigma_N=4$ m/s. These parameters
are illustrative assumptions, not estimates from the single ERA5 snapshot.
The gust is spatially uniform in this example. The aircraft's current state
and current gust are observed at each replanning time. Conditional on that
observation, the future gust has mean $e^{-\Delta/\tau}g_t$ and increasing
conditional variance. Augmenting the aircraft state with $g_t$ makes this a
fully observed Markov model. There is no observation filter in the example.

### Expected cost under several wind futures

A candidate flight can do well under one wind and poorly under another.
Suppose two equally likely winds give the following costs:

| Candidate | First wind | Second wind | Expected cost |
| --- | ---: | ---: | ---: |
| A | $0$ | $20$ | $10$ |
| B | $8$ | $8$ | $8$ |

Expected cost favors B. At $\lambda=1$, however, averaging exponential
weights gives A the score $(1+e^{-20})/2\approx0.5$, while B receives
$e^{-8}\approx0.00034$. A single favorable wind dominates A's weight.
Averaging such weights would therefore optimize a different criterion.
Exercise @ex-mppi-aircraft-expected-cost develops this distinction further.

For the intended expected-cost objective, first average costs across wind
futures. If $z^{(i)}$ specifies a candidate flight and $W^{(\ell)}$ is one
of $L$ futures conditional on the observed gust, use

$$
\widehat C(z^{(i)})=\frac1L\sum_{\ell=1}^L
 C\bigl(z^{(i)},W^{(\ell)}\bigr).
$$

Every candidate receives the same ensemble of wind futures. Pairing the
winds makes each cost difference compare two plans under the same
conditions, which can reduce variance in their estimated difference.
The ensemble is independent of the future wind that will actually occur.

The cost $C$ is fuel burned, plus quadratic penalties for horizontal and
vertical arrival error and for envelope violations. Arrival error is scaled
by tolerances of $1000$ m horizontally and $30$ m vertically. A squared unit
of either scaled arrival error adds $80$ kg-equivalent cost; envelope
violations receive a larger penalty. The code records the physical errors
separately from this combined objective.

### Smooth flight parameters and their weights

The aircraft has three control channels over a full flight. Perturbing each
short interval independently would produce a large and irregular control
sequence. Instead, a candidate is a vector $z\in\mathbb R^{13}$: one
duration parameter and four interior knots for each of the three control channels. The knots define small,
piecewise-linear perturbations around a geometric reference flight. The
nominal reference accounts for the observed state and conditional mean wind;
its forward simulation remains responsible for determining the actual
endpoint. The duration is multiplied by $e^{0.025z_0}$ and limited to
$[0.90,1.12]$ times the nominal remaining duration. This limited family keeps
the first example computationally small. It does not span all feasible
A320 trajectories.

The vertical-speed perturbation has zero time integral, so it preserves the
reference altitude change. This is a restriction on the control
parameterization. The simulator never projects the final aircraft position
onto the destination. Mach and heading perturbations can change the arrival
position, and every proposal must be checked after simulation.

Use a reference $p_0=\mathcal N(0,0.7^2I)$ and proposal
$q_\mu=\mathcal N(\mu,0.7^2I)$ over the knot and duration parameters $z$.
The weights and proposed mean are

$$
w_i\propto
\exp\!\left[-\frac{\widehat C(z^{(i)})}{\lambda}\right]
\frac{p_0(z^{(i)})}{q_\mu(z^{(i)})},
\qquad \mu^+=\sum_iw_i z^{(i)},\qquad \lambda=25\ \text{kg-equivalent}.
$$

The likelihood ratio is computed before the deterministic mapping from
parameters to bounded duration and control tapes. Clipping the duration does
not turn its physical distribution into a Gaussian; the Gaussian density
remains a density over $z$. The proposal correction appears once per
candidate, after disturbance averaging.

There are now two sample counts with different roles: $K$ candidate
parameter vectors approximate the weighted mean, while $L$ wind futures
approximate each candidate's expected cost. The parameter reference
regularizes the flight changes. It does not describe the physical gusts,
which affect ground motion and have their own temporal dependence.

The update uses the distributional construction from
@sec-mppi-distribution, with $\widehat C$ as its score. Even if expected
costs were evaluated exactly, fitting a Gaussian mean would not generally
solve the deterministic optimization over flight parameters. The finite
sampling budget introduces another approximation, so the proposed plan
must be checked before execution.

### Accepting and executing a plan

Prediction uses 60 s control intervals. Before accepting a weighted mean,
the code constructs its control tape and replays it on a 10 s grid under the
conditional mean wind. It checks the envelope and both arrival tolerances.
The proposed plan must also improve the estimated cost on the shared wind
ensemble. The best accepted plan so far is the incumbent. An invalid or
non-improving update leaves that validated plan in place. The initial geometric reference must pass the same validation. At a later
decision, the remaining previous plan is also replayed from the observed
state and can be retained as the incumbent. Validation includes the change
from the last executed command to the new first command. If neither
reference is feasible, the simulation reports a planning failure.

These checks certify feasibility only under the conditional mean and at the
checked resolution. The scenario evaluations expose possible arrival and
constraint violations under other winds. Neither a weighted average nor a
finite collection of successful scenarios guarantees feasibility for all
Gaussian disturbances.

### Comparing frozen and replanned flights

The experiment asks two separate questions: does reacting to observed wind
improve arrival, and does sampling uncertain future wind improve on its
conditional mean? Three execution strategies use paired physical wind
realizations:

1. **Frozen plan:** compute a full-trip plan using mean wind at departure and
   execute it for the rest of the flight.
2. **Conditional-mean replanning:** observe state and gust every 60 s, optimize
   the remaining trip using the conditional mean gust, and execute the next
   minute of commands.
3. **Stochastic replanning:** use the same observations and execution schedule,
   but average each candidate's cost over a conditional ensemble of wind
   futures before updating its parameters.

The saved mean-wind flight is the validated deterministic reference used to
construct the frozen plan. The two replanning variants build a geometric reference from the measured
state and compare it with the remaining previous plan, then perform two
Gaussian mean updates. They reuse the previous flight as a candidate incumbent,
while resetting the perturbation coordinates around the new geometric
reference. Within one decision, each accepted mean initializes the next
sampling update. Extending the shifted-sequence warm start in @mppi to reuse
the previous knot coordinates is a further computational experiment.

The realized gust trajectory has its own random-number stream. The optimizer
receives only the current observed gust and generates independent conditional
forecasts. All candidates at one decision share the forecast ensemble, which
reduces noise in their cost differences. Controllers encounter the same
underlying realized gust function even when their durations differ. Realized
gusts are held over each 10 s execution interval; a 5 s replay checks numerical
integration while preserving those realized inputs.

:::{figure} _static/aircraft_mppi/flight_comparison.svg
:label: fig-aircraft-mppi-profiles
:width: 100%

Complete flight profiles for the first paired gust realization. The frozen
plan executes the same commands as the mean-wind reference, so both burn the
same fuel in this model, but their ground positions differ. Replanning changes
commands in response to the observed position and gust. The horizontal line
in the final-approach panel marks the 1 km arrival tolerance.
:::

### Arrival, fuel, and numerical sensitivity

In the three recorded wind realizations, the frozen plan misses the arrival
tolerance each time. Both replanning methods arrive within tolerance in all
three trials. Their fuel difference is smaller than the distinction between
arriving and missing the destination, so arrival and fuel are reported
separately:

:::{include} artifacts/aircraft_mppi/results.md
:::

:::{figure} _static/aircraft_mppi/metrics.svg
:label: fig-aircraft-mppi-metrics
:width: 100%

Individual outcomes across three independent gust realizations, with mean and
sample-standard-deviation bars. The small trial count supports inspection of
these examples; it does not establish a reliable fuel advantage or a failure
probability. Tiny residuals near machine precision arise from floating-point
integration at the terminal altitude boundary.
:::

Replanning compensates for displacement accumulated under the frozen plan.
Whether sampling future gusts also improves fuel consumption depends on the
wind model, the trajectory family, and the sampling budget. Conditional-mean
replanning is therefore a substantive baseline. A favorable single run is
insufficient evidence that the stochastic planner is better.

:::{figure} _static/aircraft_mppi/wind_uncertainty.svg
:label: fig-aircraft-mppi-uncertainty
:width: 100%

Left: the predicted terminal-error 10th–90th percentile range from the current
planning ensemble. The conditional-mean planner has one deterministic future,
so its band has zero width. Right: conditional scenario endpoints at one
replanning time, with the 1 km arrival region. These small-ensemble ranges are
forecast diagnostics, not calibrated confidence regions or constraints.
:::

Increasing the number of candidate samples and increasing the number of
wind scenarios address different approximations. Candidate counts of $16,32,64,128$ are compared at
$L=6$; scenario counts of $1,3,6,12,24$ are compared at $K=32$. Four independent
planning seeds are used at the initial full-trip decision. Each resulting
plan is evaluated on 128 fresh, shared wind futures that no optimizer saw.
This separates variation in the candidate search from variation in the
estimated expected cost. Two projected updates need not improve monotonically
with either sample count, and this first-decision check does not establish
convergence of the entire feedback policy.

The endpoint comparisons in the recorded sweep are:

| Varied budget | Fixed budget | Mean held-out cost at the smaller budget | Mean held-out cost at the larger budget |
| --- | --- | ---: | ---: |
| $K:16\to128$ | $L=6$ | $57{,}396$ | $50{,}110$ |
| $L:1\to24$ | $K=32$ | $58{,}353$ | $51{,}319$ |

Costs are in kg-equivalent and include large penalties for rare unsuccessful
wind futures. They are not fuel totals. In the candidate-count comparison,
sample standard deviations across planning seeds are $10{,}008$ and $7{,}338$.
Intermediate budgets give nonmonotone results, so these endpoints show
sensitivity to sampling budgets without establishing convergence.

The {download}`flight diagnostics <artifacts/aircraft_mppi/metrics.json>`
include fuel, arrival errors, fine-grid residuals, effective sample sizes,
accepted and rejected updates, runtimes, and the sampling study. The local
replay displays the corresponding paths and observations. To extend the
experiment, increase the number of independent realized winds before drawing
conclusions about reliability, and compare the gain from a richer control
parameterization with the gain from more samples.

:::{iframe} ../interactive/aircraft-mppi.html
:width: 100%
:title: Montréal–Toronto stochastic flight replay
:class: mppi-replay

Select a controller and replay the recorded flight. The displayed forecast
belongs to the most recent planning decision, while the gust display follows
the observed realized wind.
:::

## Summary and Outlook

MPPI begins with the single-shooting rollout and changes how its candidate
inputs are combined. A reference distribution favors small inputs;
exponential state-cost weights favor useful trajectories; a Gaussian fit
turns the resulting distribution into a stored control sequence. Importance
correction preserves the target when samples are drawn around a nonzero
nominal sequence.

The aircraft application separates control exploration from wind uncertainty
and averages wind costs before assigning a candidate its weight. Increasing
the number of candidate controls and increasing the number of wind futures
address different estimation errors. Neither guarantees that a weighted
mean is feasible or improves the cost, so candidate updates are validated
before they replace the stored plan.

The controller still needs an execution rule. Applying one action, observing
the result, and solving again provides feedback even though each stored
sequence is open loop. [Receding-horizon control](receding-horizon-control.md)
develops that rule further, including the choice of horizon, terminal
conditions, and feasibility safeguards. These questions apply whether the
planner uses derivatives or sampled trajectories.

The method's name also reflects a connection to expectations over entire
stochastic paths. [Path-integral stochastic control](path-integral-stochastic-control.md),
after stochastic dynamic programming, develops that connection and states
the additional assumptions under which path weighting yields an optimal
current feedback action.

## Computational Sources

The aircraft experiments use small numerical modules and read precomputed
results when the book is built. The evaluator averages each candidate's
costs over a shared ensemble of possible wind futures.

- {download}`Gaussian importance weights and mean updates <code/mppi_control.py>`
- {download}`Aircraft model, wind scenarios, and trajectory optimizer <code/aircraft_mppi.py>`
- {download}`Aircraft experiment and figure generator <scripts/build_aircraft_mppi_artifacts.py>`
- {download}`ERA5 numerical snapshot <data/aircraft/era5_wind.npz>` and
  {download}`provenance and checksums <data/aircraft/era5_wind.json>`

The experiment scripts record the physical parameters, planner settings,
seeds, and validation diagnostics alongside their results. The
[aircraft replay](interactive/aircraft-mppi.html) displays recorded simulations
without a Python kernel. Regeneration commands are in the repository README.

## Exercises

:::{exercise} Weights and control updates
:label: ex-mppi-weights

An MPPI iteration has scalar nominal control $u=0$, perturbations
$(-1,0,1)$, corrected trajectory costs $(\log4,\log2,0)$, and $\lambda=1$.

**(a)** Compute the normalized weights and the updated control.

**(b)** Add $1000$ to every cost. Explain why the mathematical update is
unchanged and how to compute it without numerical underflow.

**(c)** Keep the corrected costs fixed and let $\lambda$ approach zero.
Which perturbation determines the limiting update?
:::

:::{solution} ex-mppi-weights
:class: dropdown

**(a)** The unnormalized weights are $(1/4,1/2,1)$, whose sum is $7/4$.
The normalized weights are $(1/7,2/7,4/7)$, giving
$u^+=-(1/7)+(4/7)=3/7$.

**(b)** The common factor $e^{-1000}$ cancels between numerator and
denominator. Direct evaluation would underflow in ordinary floating-point
arithmetic. Subtracting the minimum corrected cost before exponentiating
recovers the original three weights.

**(c)** The unique minimum cost belongs to perturbation $1$. Its normalized
weight approaches one, so $u^+\to1$. The qualification that corrected costs
are fixed matters: a likelihood correction can itself depend on $\lambda$.
:::

:::{exercise} The likelihood correction
:label: ex-mppi-likelihood-correction

Consider the deterministic rollout $x_{k+1}=x_k+hv_k$ with $x_0=0$,
target $a=1$, $N=10$ intervals of length $h=0.1$, and terminal cost
$\tfrac\kappa2(x_N-a)^2$ with $\kappa=4$. Set $\lambda=0.5$ and
choose a Gaussian reference covariance $\Sigma_h=\lambda/(rh)$ with
$r=1$, so its KL penalty gives the quadratic mean-control effort.
The proposal mean is $u_k=0.4$ at every step.

**(a)** Complete the square in the Gibbs density to find its mean. You
may use the fact that the mean of a Gaussian equals the minimizer of the
quadratic in its negative log density.

**(b)** Find the mean estimated when the weights include the state cost
but omit the reference-to-proposal density ratio.

**(c)** Implement both estimators and compare their mean control, averaged
over the ten time slots, as the sample count increases. Does increasing
the sample count correct the discrepancy in part (b)?
:::

:::{solution} ex-mppi-likelihood-correction
:class: dropdown

**(a)** Up to a constant, the negative log Gibbs density is

$$
\frac{rh}{2\lambda}\sum_kv_k^2
+\frac\kappa{2\lambda}\left(h\sum_kv_k-a\right)^2.
$$

Its first-order condition is
$r v_k+\kappa(h\sum_jv_j-a)=0$. All coordinates of its minimizer are
equal, giving

$$
\mathbb E_{P^\star}v_k=\frac{\kappa a}{r+\kappa Nh}=0.8.
$$

The result is independent of the proposal mean because the likelihood
ratio cancels the proposal density in the expectation.

**(b)** Omitting that ratio replaces the reference quadratic
$\sum_kv_k^2$ by $\sum_k(v_k-0.4)^2$. The corresponding first-order
condition is $r(v_k-0.4)+\kappa(h\sum_jv_j-a)=0$, so its mean is

$$
\frac{0.4r+\kappa a}{r+\kappa Nh}=0.88.
$$

Thus the two procedures converge to different expectations.

**(c)** A minimal implementation uses the same perturbations for both
estimates so that their difference is easier to observe:

```python
import numpy as np

rng = np.random.default_rng(728)
N, h, r, kappa, lam = 10, 0.1, 1.0, 4.0, 0.5
U = np.full(N, 0.4)
for K in (1_000, 10_000, 100_000):
    eps = rng.normal(size=(K, N)) * np.sqrt(lam / (r * h))
    inputs = U + eps
    cost = 0.5 * kappa * (h * inputs.sum(axis=1) - 1.0) ** 2
    corrected = cost + h * r * (eps @ U)
    estimates = []
    for scores in (corrected, cost):
        weights = np.exp(-(scores - scores.min()) / lam)
        weights /= weights.sum()
        estimates.append((weights @ inputs).mean())
    print(K, *estimates)
```

The estimates approach $0.8$ and $0.88$. More samples reduce Monte Carlo
error; they do not repair an omitted likelihood ratio. Individual runs
need not approach their limits monotonically.
:::

:::{exercise} Stochastic aircraft rollouts and expected cost
:label: ex-mppi-aircraft-expected-cost

An aircraft simulator accepts a control sequence $\mathbf U$ and a gust
history $W$, and returns cost $C(\mathbf U,W)$. The objective is
$\mathbb E_W C(\mathbf U,W)$.

**(a)** A sampling optimizer generates one gust history per candidate
control sequence and assigns weight proportional to
$e^{-C(\mathbf U,W)/\lambda}$. Explain why collecting more such weighted
samples does not generally recover the intended expected-cost objective.

**(b)** Consider a safe candidate with constant cost $1$ and a risky
candidate with cost $0$ with probability $0.9$ and $20$ with probability
$0.1$. At $\lambda=1$, compare their expected costs and their exponential
scores $-\log\mathbb E[e^{-C}]$.

**(c)** Describe a sampling design that ranks candidate controls by an
estimate of expected cost. Does this design guarantee that the weighted
mean is optimal or feasible?
:::

:::{solution} ex-mppi-aircraft-expected-cost
:class: dropdown

**(a)** For a fixed candidate, averaging the weights estimates
$\mathbb E_W[e^{-C/\lambda}]$, whereas the desired score depends on
$\mathbb E_W C$. The exponential is nonlinear. Favorable gust histories
receive disproportionate weight, so this procedure can favor a
candidate whose expected cost is larger. The reference-to-proposal
correction for control sampling does not remove this difference in the
disturbance objective.

**(b)** The safe candidate has expected cost $1$ and exponential score
$1$. The risky candidate has expected cost $2$ but exponential score
$-\log(0.9+0.1e^{-20})\approx0.105$. Expected cost favors the safe
candidate; the exponential score favors the risky candidate.

**(c)** For each candidate, simulate multiple gust histories and form
$\widehat C(\mathbf U)=L^{-1}\sum_{\ell=1}^L C(\mathbf U,W^{(\ell)})$.
Then use $\widehat C$ as the candidate score in the sampling optimizer.
Shared gust histories across candidates can make paired comparisons
less variable. As the number of disturbance samples increases, this
score estimates the intended expected cost. The subsequent Gaussian mean
fit need not minimize that cost over deterministic plans. Finite sampling
introduces further error, and an average of feasible plans may violate
nonconvex constraints. The proposed plan still needs cost and feasibility
checks.
:::

:::{exercise} Shifting a plan and checking its mean
:label: ex-mppi-execution

A controller stores $(u_0,u_1,u_2)=(0.6,0.4,0.2)$ for three intervals.
After the first interval it observes a new state.

**(a)** Which control was executed, and what sequence initializes the next
planning problem if the final control is repeated? What other input to the
rollout changes at this observation?

**(b)** Consider the separate one-step system $x_1=u$ with terminal
constraint $x_1\in[-1.1,-0.9]\cup[0.9,1.1]$. Two feasible candidates are
$u=-1$ and $u=1$. Is their equally weighted mean feasible? What should an
implementation with a validated incumbent do if this mean is proposed?
:::

:::{solution} ex-mppi-execution
:class: dropdown

**(a)** The controller executed $0.6$. Its shifted initial guess is
$(0.4,0.2,0.2)$, and the measured state becomes the initial state for every
new rollout. The guess is then updated by sampling; it is not a commitment
to execute all three entries.

**(b)** The mean is $u=0$, which produces $x_1=0$ outside both allowed
intervals. Feasible samples do not ensure a feasible mean under nonconvex
constraints. The implementation should validate the proposal and retain
the feasible incumbent when that validation fails, as in the aircraft
example.
:::

## Self-checks

:::{exercise} Sampling and feedback
:label: ex-mppi-check-feedback

What makes the executed controller a feedback controller: drawing many
trajectories, or conditioning a new plan on the next observed state?
:::

:::{solution} ex-mppi-check-feedback
:class: dropdown

Conditioning a new plan on the observed state supplies feedback. A control
sequence computed from many samples and then frozen still executes open loop.
:::
