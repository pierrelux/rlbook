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
candidate flight's expected cost before assigning its weight. A reactor
example applies the same update to temperature and composition fields
governed by a partial differential equation, first planning with the
equation itself and then with a learned surrogate.
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
- Reduce a field-control problem to a few bounded sampling parameters, and
  validate each sampled update in the full simulation before it replaces a
  feasible plan.

## Prerequisites

The algorithm uses the [single-shooting rollout](numerical-trajectory-optimization.md)
and the [finite-horizon optimal-control formulation](discrete-time-optimal-control.md).
The sampling derivation requires Gaussian distributions and expectations;
KL divergence and importance sampling are developed here. The aircraft
example uses conditional disturbance models from
[Stochastic Dynamics and Partial Observation](stochastic-dynamics-observation.md).
The reactor example states the finite-volume model it uses and assumes no
background in partial differential equations.

(sec-mppi)=
## Improving a Control Sequence by Sampling

We already know how to evaluate a control sequence: simulate the system
and compute the resulting cost. How can we improve that sequence? Instead
of computing a gradient, we try many random variations, evaluate each one,
and form a weighted average that gives more influence to the lower-cost
candidates. That average becomes our next control sequence.

If the system has $d_u$ control components, then
$\mathbf u_k\in\mathbb R^{d_u}$ contains the inputs applied together at
time step $k$. The current plan
$\mathbf U=(\mathbf u_0,\ldots,\mathbf u_{N-1})$ collects these vectors
over all $N$ steps. Thus $\mathbf u_k$ is one step's control vector, while
$\mathbf U$ is the complete sequence, with $Nd_u$ entries when stacked
into one vector.

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
using the update in the aircraft and reactor examples.

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
$\mathbf{x}_{k+1}=\mathbf{f}_k(\mathbf{x}_k,\mathbf{v}_k)$, where
$\mathbf v_k\in\mathbb R^{d_u}$ is the candidate control vector at step
$k$. Collect these vectors into the complete candidate sequence
$\mathbf V=(\mathbf v_0,\ldots,\mathbf v_{N-1})$, with the same structure
as the nominal sequence $\mathbf U$. The initial state is
fixed, so each input sequence determines a simulated state sequence.
Define the state and terminal cost of a rollout by

$$
S(\mathbf V)=c_N(\mathbf{x}_N)
+\sum_{k=0}^{N-1}c_k(\mathbf{x}_k).
$$

Each $c_k$ is the cost for one interval; if it comes from a continuous
running cost, its discretization includes the interval length.

In the earlier optimal-control formulation, a stage cost
$c_k(\mathbf x_k,\mathbf u_k)$ could penalize both the state and the
control effort. Here $c_k(\mathbf x_k)$ contains only the state cost;
the inputs still affect $S(\mathbf V)$ through the simulated states.
Control effort will be accounted for separately by a KL-divergence
penalty between the input distribution and a reference distribution.

Choose a Gaussian reference distribution $P_0$ centered at zero to express
a preference for small inputs. As derived below, this choice makes the KL
penalty a quadratic cost on the magnitude of the nominal controls; the
penalty vanishes when the nominal sequence is zero. Use a separate Gaussian
sampling distribution $Q_{\mathbf U}$ over complete input sequences,
centered at the current plan $\mathbf U$. At each step $k$, the candidate
control $\mathbf v_k$ has mean $\mathbf0$ under $P_0$ and mean
$\mathbf u_k$ under $Q_{\mathbf U}$. Both use the same positive-definite
covariance $\Sigma\in\mathbb R^{d_u\times d_u}$ for the control components
at a single step. Assuming independence across time gives the joint
densities over the complete sequence:

$$
p_0(\mathbf V)=\prod_{k=0}^{N-1}\mathcal N(\mathbf v_k;\mathbf0,\Sigma),
\qquad
q_{\mathbf U}(\mathbf V)=
\prod_{k=0}^{N-1}\mathcal N(\mathbf v_k;\mathbf u_k,\Sigma).
$$

Centering the reference at zero is a modeling choice. A Gaussian reference
could instead be centered on a prescribed baseline sequence; the KL penalty
would then charge for squared deviations of the nominal controls from that
baseline. In either case, the state and terminal costs can favor controls
away from the reference mean when their benefits justify the effort.

The reference remains fixed during a planning update.
The sampling mean is the entire nominal sequence $\mathbf U$. This mean
is taken over repeated candidate draws, with a separate mean control
vector $\mathbf u_k$ for each time step. Changing $\mathbf u_k$ moves
the center of the candidate controls at step $k$.
Initializing $\mathbf U=\mathbf0$ means setting every
$\mathbf u_k=\mathbf0$, so the sampling and reference distributions
initially coincide. Later updates can change the nominal controls at each
step, moving the samples toward a new plan while the reference stays
centered at zero. The sampler can therefore change where it searches
without changing the reference that defines the objective.

### Exponentially weighting the reference

Sampling from the reference determines how often different input sequences
appear in a batch. Assigning a cost weight to each draw then changes how
much that occurrence contributes to the weighted average. The total weight
assigned to a sequence depends on both its sampling frequency and its
weight per occurrence, which is why these two factors multiply.

For example, first consider a reference with only two possible input
sequences, $a$ and $b$, selected with probabilities $3/4$ and $1/4$.
Suppose their costs are $S(a)=\lambda\log3$ and $S(b)=0$, so each draw of
$a$ receives score $1/3$ and each draw of $b$ receives score $1$. Their
reference probabilities multiplied by these scores are

$$
\frac34\times\frac13=\frac14
\quad\text{for }a,
\qquad
\frac14\times1=\frac14
\quad\text{for }b.
$$

Dividing by the sum, $1/2$, gives probability $1/2$ to each sequence.
The reference samples $a$ three times as often, but each occurrence
receives one third of the weight, so the two effects cancel.

For continuous inputs, the same operation multiplies the reference density
$p_0(\mathbf V)$ by the score $e^{-S(\mathbf V)/\lambda}$ at each sequence
$\mathbf V$, then divides by the integral of that product:

$$
\begin{aligned}
p^\star(\mathbf V)
&=\frac{e^{-S(\mathbf V)/\lambda}p_0(\mathbf V)}{Z},
\qquad \lambda>0,\\
Z&=\int e^{-S(\mathbf V)/\lambda}p_0(\mathbf V)\,d\mathbf V
=\mathbb E_{P_0}[e^{-S(\mathbf V)/\lambda}].
\end{aligned}
$$

The score $e^{-S(\mathbf V)/\lambda}$ is not itself a probability density,
and the product does not define a joint distribution of two random
variables. Both factors concern the same input sequence: $p_0$ specifies
its density before costs are considered, and the score adjusts that density
according to its cost. The normalizing constant $Z$ makes the resulting
density integrate to one.

Assume $S$ is measurable and bounded below and $0<Z<\infty$.
The normalized result is a **Gibbs distribution**: two trajectories with
equal reference density and cost difference $\Delta S$ have density ratio
$e^{-\Delta S/\lambda}$. The parameter $\lambda$ has the units of cost
and controls how strongly those costs affect the ratio.

The opening batch weights contain only the scores because drawing from
$P_0$ already accounts for the reference probabilities. Multiplication by
$p_0$ appears when we describe the distribution represented by those
weighted samples. This explains how weighting changes a distribution;
the particular choice of an exponential score is justified by the
objective combining expected cost and KL divergence below.

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
=\mathbb E_{\mathbf V\sim Q}[S(\mathbf V)]
+\lambda D_{\mathrm{KL}}(Q\|P_0).
```

Here $Q$ ranges over distributions on complete input sequences for which
both terms are finite; it is not restricted to a Gaussian family.

Recall that $P^\star$ is the distribution obtained by multiplying the
reference density by the exponential cost score and normalizing. Its
density is $p^\star(\mathbf V)=e^{-S(\mathbf V)/\lambda}p_0(\mathbf V)/Z$.
To verify that this distribution solves the optimization problem, we can
express the difference between $\mathcal J(Q)$ and its minimum as a
nonnegative divergence. Substituting this density into the logarithm gives

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

The quantity $Z$ is fixed by $S$, $P_0$, and $\lambda$, so the term
$-\lambda\log Z$ does not depend on $Q$. Choosing $Q=P^\star$ makes the
remaining term, $\lambda D_{\mathrm{KL}}(Q\|P^\star)$, equal to zero.
Every other distribution gives a positive divergence and hence a larger
objective value. This proves that
$P^\star=\operatorname*{arg\,min}_Q\mathcal J(Q)$, with minimum value
$\mathcal J(P^\star)=-\lambda\log Z$.

The exponential cost weight is contained in the definition of $P^\star$;
it is no longer explicit in the KL identity because we took logarithms.
Writing out the density of the minimizing distribution gives

$$
p^\star(\mathbf V)
=\frac{1}{Z}\,
\underbrace{e^{-S(\mathbf V)/\lambda}}_{\text{exponential cost weight}}
\,p_0(\mathbf V).
$$

Multiplying the reference density $p_0(\mathbf V)$ by the indicated factor
and normalizing by $Z$ is the exponential weighting introduced above.
The bound on $\mathcal J(Q)$ proves that the distribution constructed this
way solves @eq-mppi-distribution-objective.

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

Only the mean sequence is adjustable; the covariance $\Sigma$ at each
step remains fixed.
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

With an exact solution of @eq-mppi-gaussian-objective, the updated Gaussian
would have the same mean as $P^\star$. The equation above identifies that
mean as the quantity we need to compute; it does not yet provide a way to
evaluate it. We know the density of $P^\star$ up to its normalizing constant:
for any candidate sequence $\mathbf V$, a rollout gives $S(\mathbf V)$,
so we can evaluate $e^{-S(\mathbf V)/\lambda}p_0(\mathbf V)$. In general,
however, calculating $Z$ or $\mathbb E_{P^\star}[\mathbf v_k]$ requires
high-dimensional integration, and we do not have a direct sampler for
$P^\star$.

Importance sampling in @sec-mppi-importance will estimate this mean from
candidates drawn from the Gaussian $Q_{\mathbf U}$, which we can sample.
The normalized weights will account for the difference between the two
distributions without requiring the value of $Z$. Thus the computed update
will only approximate the mean of $P^\star$; the equality above describes
the exact target of the calculation. The quadratic example below first
provides a case where this target can be calculated analytically.

Even if its mean were computed exactly, the Gaussian would retain covariance
$\Sigma$ at each step. It can therefore lose features of $P^\star$: if
low-cost sequences form separate groups, their average can lie between
those groups.

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

(sec-mppi-importance)=
### Correcting for where the samples were drawn

The mean $\mathbf u_k^+=\mathbb E_{P^\star}[\mathbf v_k]$ is the exact
solution of the Gaussian approximation problem. To estimate it for a
general rollout, we have two available operations: draw a sequence from
$Q_{\mathbf U}$, and evaluate its cost $S(\mathbf V)$. We can use these
operations without a direct sampler for $P^\star$ or an analytic value
for its normalizing constant $Z$.

Substituting the definition of $p^\star$ into the desired expectation and
writing $Z$ as its defining integral gives

$$
\mathbf u_k^+
=\frac{\int \mathbf v_k e^{-S(\mathbf V)/\lambda}p_0(\mathbf V)\,d\mathbf V}
{\int e^{-S(\mathbf V)/\lambda}p_0(\mathbf V)\,d\mathbf V}.
$$

Both integrals involve the unnormalized density that we can evaluate.
Multiplying and dividing their integrands by $q_{\mathbf U}(\mathbf V)$
expresses them as expectations under the distribution we can sample:

$$
\mathbf u_k^+
=\frac{\mathbb E_{Q_{\mathbf U}}
\left[e^{-S(\mathbf V)/\lambda}
\frac{p_0(\mathbf V)}{q_{\mathbf U}(\mathbf V)}\mathbf v_k\right]}
{\mathbb E_{Q_{\mathbf U}}
\left[e^{-S(\mathbf V)/\lambda}
\frac{p_0(\mathbf V)}{q_{\mathbf U}(\mathbf V)}\right]}.
$$

This change of sampling distribution is **importance sampling**. A batch
from $Q_{\mathbf U}$ can estimate both expectations, using the score
$e^{-S(\mathbf V)/\lambda}p_0(\mathbf V)/q_{\mathbf U}(\mathbf V)$ for each
candidate. Dividing by the sum of these scores normalizes the sample
weights, so we do not need to know $Z$.

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

The ratio of expectations above is exact. To compute an approximate update,
replace both expectations by averages over $K$ independent samples from
$Q_{\mathbf U}$. For numerical stability, subtract
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

(sec-reactor-mppi)=
## Heater Control of a Flow-Through Reactor

The aircraft state contained a few physical quantities and a gust. Many
process systems are instead described by fields: the temperature and
composition at every point of a vessel. In a flash calciner, clay particles
carried by hot gas are heated rapidly while an endothermic reaction converts
them into a reactive material that can replace part of the clinker in cement
{cite:p}`Cantisani2024FlashCalciner`. Heaters act on broad regions of such a
reactor, while the quantities that matter, such as the conversion of the
material leaving it and the highest local temperature, depend on how heat is
distributed along the flow.

MPPI needs only the ability to simulate a candidate control sequence and
evaluate its cost, and a discretized partial differential equation (PDE)
provides both. The number of state values does not enter the update
directly: the weights depend on candidate costs, and the update averages
control parameters. The state dimension determines the cost of each rollout
and how much of the state a few actuators can influence. This section
controls a reactor with 4,096 state values through 24 sampled parameters,
first planning with the PDE itself and then with a learned surrogate.

### A two-field reactor model

Material flows from left to right through a rectangular domain of length
$8$ m and width $2$ m, with $x$ measured downstream and $y$ across the width.
Two fields describe its state: temperature $T(x,y,t)$ in kelvin and remaining
reactant fraction $c(x,y,t)\in[0,1]$. Conversion is $a=1-c$. Both fields are
carried by a prescribed downstream velocity $v_x(y)$, spread by diffusion,
and coupled through a temperature-dependent reaction:

$$
\begin{aligned}
\partial_t c+v_x(y)\,\partial_x c&=D\nabla^2c-k(T)\,c,\\
\partial_t T+v_x(y)\,\partial_x T&=D\nabla^2T-\beta(T-T_{\mathrm{wall}})
-\Delta T_{\mathrm{rxn}}\,k(T)\,c
+q_{\max}\sum_{j=1}^4u_j\,\mathbb 1_{\Omega_j}(x,y).
\end{aligned}
$$

The reaction rate has the Arrhenius form
$k(T)=0.1\exp[12000(1/950-1/T)]$ s$^{-1}$. Converting a unit of reactant
cools the material by $\Delta T_{\mathrm{rxn}}$, and heat is lost toward the
wall temperature at rate $\beta$. Each heater covers one quadrant $\Omega_j$
of the domain, and its input $u_j\in[0,1]$ is a fraction of the maximum
heating rate $q_{\max}$. Heater H1 covers the upstream half of the bottom
($y<1$ m), H2 the downstream half of the bottom, H3 the upstream half of
the top, and H4 the downstream half of the top. Material enters with
reactant fraction one.

| Quantity | Value |
| --- | --- |
| Domain and mesh | $8$ m $\times$ $2$ m, $64\times32$ cells |
| Diffusivity $D$ of both fields | $5\times10^{-4}$ m$^2$/s |
| Heat-loss rate $\beta$ and wall temperature | $0.01$ s$^{-1}$, $700$ K |
| Reaction cooling $\Delta T_{\mathrm{rxn}}$ | $60$ K per unit of reactant |
| Maximum heating rate $q_{\max}$ | $20$ K/s |
| Inlet temperature | $800+10\cos(\pi y/2)$ K |
| Operating limits | $\max T\leq1070$ K, outlet conversion $\geq95\%$ |

These are teaching assumptions inspired by flash calcination, not calibrated
calciner parameters. The model omits separate solid and gas phases,
detailed chemistry, residence-time distributions, and industrial geometry.

The Arrhenius factor makes conversion sensitive to temperature. At $950$ K
the rate is $0.1$ s$^{-1}$, a time constant of $10$ s. At $900$ K it falls
to about $0.050$ s$^{-1}$, and at $1000$ K it rises to about
$0.19$ s$^{-1}$. Material spends about $40$ s in the reactor, so a parcel
that remains $50$ K colder than its surroundings for part of that time
leaves with noticeably more unreacted material. More heating raises the
rate, but it also raises the peak temperature, which must stay below
$1070$ K everywhere. Because the flow speed can vary across the width, the
outlet conversion is weighted by flux,

$$
a_{\mathrm{out}}=1-\frac{\sum_i v_x(y_i)\,c(8,y_i)}{\sum_i v_x(y_i)},
$$

where the sums run over the outlet cells. It must remain at least $95\%$.

A finite-volume discretization with $64\times32$ cells turns the PDE into
$2\times64\times32=4{,}096$ ordinary differential equations. Advection uses
first-order upwind fluxes; diffusion uses centered fluxes with no diffusive
flux through the boundaries. An explicit two-stage strong-stability-preserving
Runge–Kutta method advances time in substeps of at most $0.1$ s. Heater
inputs are held for $\Delta t=2$ s, and integrating over one control
interval defines the transition

$$
X_{k+1}=F(X_k,u_k),\qquad X_k\in\mathbb R^{4096},\quad u_k\in[0,1]^4,
$$

which plays the role of the rollout map $\mathbf f_k$ in @mppi. The mesh
partitions physical space; it is not a grid over the 4,096-dimensional
state space. Refining it makes each rollout more expensive without changing
the number of heater inputs.

Two flow profiles appear below, both with mean speed $0.2$ m/s. The
field-recovery problem uses a shear flow that is fastest at the center of
the channel, $v_x(y)=0.16+0.24\,\eta(1-\eta)$ m/s with $\eta=y/2$. Material
then takes about $36$ s to cross the reactor at the center and nearly $50$ s
near the walls, so a disturbance tilts and stretches as it moves. The
surrogate experiment uses uniform flow at $0.2$ m/s.

### Restoring the operating field

Holding all four heaters at $0.46$ for $400$ s under the shear flow
produces a steady operating state. Its temperature rises from about $800$ K
at the inlet to a maximum of $1063.54$ K, and its outlet conversion is about
$95.8\%$. Denote its fields by $T_\star$ and $a_\star$. Starting from this
state, a cold patch enters through the lower part of the inlet:

$$
\Delta T_{\mathrm{in}}(y,t)=-100\exp\!\left[-\frac{(y-0.55)^2}{2(0.22)^2}\right]
\sin^2\!\left(\frac{\pi(t-6)}{8}\right)\ \mathrm K,
\qquad 6\leq t\leq14\ \mathrm s,
$$

and $\Delta T_{\mathrm{in}}=0$ otherwise. Control starts at $14$ s, when the
whole patch has entered the reactor and is traveling through the bottom half
where H1 and H2 act. If the heaters stay at $0.46$, the colder material
reacts too slowly: outlet conversion falls to $94.75\%$ and remains below
the $95\%$ limit for $7.5$ s.

The task is to return both fields to the operating state while respecting
the limits. The inlet is known to have returned to its normal profile, both
fields are measured exactly at each decision, and the PDE is deterministic.
As in [the derivation of the update](#sec-mppi-distribution), randomness
enters only through the candidate controls.

Departure from the operating state is measured by the normalized field error

$$
E(X)=\frac1{2048}\sum_{\text{cells}}\left[
\left(\frac{T-T_\star}{20\ \mathrm K}\right)^2+
\left(\frac{a-a_\star}{0.1}\right)^2\right].
$$

A cell that is $20$ K too cold contributes as much as a cell whose conversion
is $0.1$ too low. The squared error penalizes warming as well as cooling, so
heat delivered to material outside the cold patch counts against a plan.
Each heater acts on a quadrant of $512$ cells, and four such inputs cannot
set the $4{,}096$ state values independently. Some error must remain until
the cold material leaves the reactor.

MPPI plans over $N=20$ intervals, a $40$ s horizon equal to the mean
residence time. With $u_{-1}$ the control applied during the previous
interval, the planning objective is

$$
J=\frac1{20}\sum_{k=0}^{19}\left[E(X_{k+1})
+\frac{0.01}{4}\sum_{j=1}^4\left(\frac{u_{k,j}-0.46}{0.05}\right)^2
+\frac{0.005}{4}\sum_{j=1}^4\left(\frac{u_{k,j}-u_{k-1,j}}{0.05}\right)^2\right]
+E(X_{20}).
$$

The second term discourages departures from nominal heating, and the third
discourages abrupt changes between intervals. Earlier in the chapter, control
effort entered only through the KL penalty and $S$ contained state costs.
Here $J$ plays the role of $S$ and includes control terms written in
physical heater units. The derivation of $P^\star$ used only that $S$
assigns a cost to each complete candidate sequence, so these terms change
the target distribution without changing the update.

The limits enter in two ways. During sampling, a candidate whose states at
the $2$ s decision times exceed $1070$ K anywhere, or bring outlet conversion
below $95\%$, receives infinite cost and therefore zero weight. In the
distributional objective, this restricts the reference to sequences that pass
the check. That set need not be convex, so the mean of the resulting Gibbs
distribution need not satisfy the limits, as in Exercise @ex-mppi-execution.
Each proposed update therefore requires its own check before it can replace
the stored plan.

### Sampling in a bounded control space

A plan contains $20\times4=80$ heater values. Sampling each independently
would produce irregular schedules and spend samples on rapid fluctuations that
the third term of $J$ penalizes. As in the aircraft example, a candidate is
described by a few smooth parameters: six knots for each heater, linearly
interpolated to the $20$ decision times. This gives $24$ sampled parameters,
collected in a $6\times4$ array $z$.

Heater inputs must stay in $[0,1]$, while Gaussian samples are unbounded.
Clipping would change the distribution of the clipped values, as noted after
@mppi. Instead, a logistic map takes latent parameters to physical knot
values,

$$
u_{\mathrm{knots}}(z)=\operatorname{sigmoid}\bigl(\operatorname{logit}(u_{\mathrm{seed}})+0.12\,z\bigr),
$$

applied elementwise. The array $u_{\mathrm{seed}}$ contains the stored plan
evaluated at the six knot times, so $z=0$ reproduces the stored plan up to
interpolation error. The derivative of the sigmoid is $u(1-u)$, which is
about $0.25$ near $u=0.46$. A latent change of $0.7$, one standard deviation
of the sampling distribution below, therefore moves a knot by about
$0.12\times0.7\times0.25\approx0.021$, or two percentage points of heating.

Gaussian densities are defined on $z$. The reference is
$p_0=\mathcal N(0,0.49I)$ and the proposal is $q_\mu=\mathcal N(\mu,0.49I)$.
At each decision the decoder and the reference stay fixed while $\mu$
starts at zero. Each of four updates draws $K=256$ candidates
$z^{(i)}\sim q_\mu$, decodes them, simulates them in the PDE, and computes

$$
w_i\propto\exp\!\left[-\frac{J(z^{(i)})}{\lambda}\right]
\frac{p_0(z^{(i)})}{q_\mu(z^{(i)})},
\qquad
\mu_{\mathrm{weighted}}=\sum_{i=1}^Kw_i\,z^{(i)},
\qquad \lambda=0.01.
$$

This is the [importance-corrected update](#sec-mppi-importance) in latent
coordinates, with $\Sigma=0.49I$ and the mean $\mu$ in place of the nominal
controls. The weighting temperature $\lambda$ is dimensionless, like $J$,
and unrelated to temperature in kelvin. Since latent zero decodes to the
stored plan, the KL penalty charges for changing that plan, not for heating
as such. This is the alternative, [noted with the reference
distribution](#sec-mppi-distribution), of centering the reference on a
baseline sequence, with the previous plan as the baseline. The average is
taken in latent coordinates, so the decoded heater schedule generally
differs from the weighted average of the candidates' physical schedules.

The temperature sets which cost differences matter. At the first decision,
the stored plan has $J=0.0488$. Before correction, a candidate whose cost
exceeds another's by $\lambda\log2\approx0.0069$, about $14\%$ of that value,
receives half the weight.

The likelihood correction changes the weights substantially once $\mu$ has
moved. At the fourth update of the first decision, after three accepted
updates, candidate 12 had cost $0.05649$ and candidate 180 had cost
$0.05840$. Exponential cost weighting alone would give candidate 12 about
$e^{0.191}\approx1.21$ times the weight of candidate 180. Its recorded weight,
$0.0682$, was instead less than half of candidate 180's weight, $0.1565$. The
ratio corresponds to corrected costs that differ by
$-\lambda\log(0.0682/0.1565)\approx0.0083$ in favor of candidate 180, so the
correction term contributes about $0.0102$. Candidate 12 lies farther in the
direction in which the proposal had already moved, where the proposal
samples more often than the reference. Exercise @ex-mppi-reactor-latent
works through this calculation.

### Validating and executing a sampled update

Each update ends by checking the proposed mean. The controller decodes
$\mu+\alpha(\mu_{\mathrm{weighted}}-\mu)$ for step fractions
$\alpha\in\{1,\tfrac12,\tfrac14,\tfrac18\}$ in turn, and accepts the first
that lowers $J$ below the stored plan's value and passes a finer check. The
finer check records the PDE state every $0.25$ s over the $40$ s plan,
followed by another $40$ s at nominal heating, and tests both limits at every
record. If no step fraction succeeds,
both $\mu$ and the stored plan remain unchanged.

After four updates, the controller applies the first $2$ s of the stored
plan to the PDE, shifts the plan by one interval, and appends nominal
heating. The stored plan is the exact heater sequence that passed
validation; its six-knot approximation $u_{\mathrm{seed}}$ only centers the
next round of sampling. Because the plant is the same deterministic PDE with
a known inlet, executing the first action reproduces the predicted state.
The shifted plan followed by nominal heating then repeats a trajectory that
has already passed the finer check, except for its final $2$ s. The
validation tail at nominal heating is what makes this continuation
available: if every later update is rejected, the controller still holds a
checked plan. [Receding-horizon control](receding-horizon-control.md)
develops this property, recursive feasibility, with terminal conditions in
place of a simulated tail.

The procedure needs a feasible plan at the first decision. Unchanged heating
does not qualify, since it violates the conversion limit. The experiment
supplies an initial schedule that holds H1 at $0.50$ for $8$ s and returns it
to $0.46$ over the following $8$ s, with the other heaters at nominal. This
schedule is given to the controller, not found by it; MPPI starts from it and
revises it at every decision.

### Recorded recovery

All three schedules start from the same disturbed state at $14$ s and are
recorded until $80$ s on a $0.25$ s output grid. The integrated field error
is $\int_{14}^{80}E(X(t))\,dt$, and heating energy is the integral of the
mean heater fraction over the same interval. Both are measured in seconds;
heating energy counts seconds at full average heating, since the model has no
heat capacity to convert it to joules.

| Schedule | Integrated field error | Heating energy | Maximum temperature | Lowest outlet conversion |
| --- | ---: | ---: | ---: | ---: |
| Unchanged heating | $3.709$ s | $30.360$ s | $1063.54$ K | $94.747\%$ |
| Supplied initial schedule, executed unchanged | $2.098$ s | $30.484$ s | $1066.94$ K | $95.271\%$ |
| MPPI | $1.962$ s | $30.439$ s | $1066.16$ K | $95.329\%$ |

The supplied schedule alone removes $43\%$ of the integrated error of
unchanged heating and restores the conversion limit. MPPI removes $47.1\%$,
which is $6.5\%$ less error than the supplied schedule, while using slightly
less heat than that schedule and staying $3.8$ K below the temperature limit.

:::{figure} _static/thermal_reactor/recovery-storyboard.svg
:label: fig-reactor-recovery-fields
:width: 100%
:alt: Four rows of reactor maps at 22, 34, and 46 seconds show temperature and conversion error relative to the operating state. A blue cold region in the lower half moves from about 2.5 meters to the outlet. Under MPPI the blue region is fainter and surrounded by faint red warming at 22 seconds.

Departures from the operating state at $22$, $34$, and $46$ s. The first two
rows show temperature error under unchanged heating and under MPPI; the last
two rows show conversion error. Blue marks colder material or lower
conversion. Dotted lines mark the heater quadrants. The deficit is carried
downstream and tilted by the shear flow. MPPI reduces it, while each heater
also warms material around the patch.
:::

The controls in @fig-reactor-recovery-control follow the patch. MPPI raises
H1 by about five percentage points while the patch crosses the upstream
bottom quadrant, then returns it to nominal by about $32$ s. Between about
$36$ and $62$ s it lowers H2, by up to $0.8$ percentage points near $48$ s,
as material warmed by H1 passes through the downstream bottom quadrant; the
squared error penalizes that excess warmth. The top heaters change by less
than one percentage point. Outlet conversion still dips as the patch leaves,
but its lowest value is $95.33\%$ instead of $94.75\%$.

:::{figure} _static/thermal_reactor/recovery-control.svg
:label: fig-reactor-recovery-control
:width: 100%
:alt: Four panels over 14 to 80 seconds. Normalized field error falls fastest for MPPI and the supplied schedule. Outlet conversion dips below 0.95 only under unchanged heating. Maximum temperature stays below the 1070 kelvin limit for all schedules. The heater panel shows H1 about 0.05 above nominal early, H2 slightly below nominal between about 36 and 62 seconds, and H3 and H4 near nominal.

Recovery from the disturbed state at $14$ s. Top left: normalized field
error $E(X(t))$. Top right: outlet conversion, with the $95\%$ limit dashed.
Bottom left: maximum temperature, with the $1070$ K limit dashed. Bottom
right: executed MPPI heater inputs relative to nominal heating $0.46$, each
held for $2$ s.
:::

The sampling statistics show how much of this result comes from accepted
updates. Over $33$ decisions, $24$ of $132$ updates were accepted. Only four
of the accepted updates used the full weighted mean; the other $20$ needed a
step fraction of $\tfrac12$ or less. Nine acceptances occurred in the first
three decisions and eleven more between $32$ and $42$ s, as the patch
approached the outlet. From $44$ s on, only $3$ of $72$ updates were
accepted. By then the stored plan was already returning to nominal heating,
and random perturbations tended to add error to a field that was recovering
on its own. The effective sample size ranged from $19.8$ to $64.1$ of $256$
candidates, and between $108$ and $217$ candidates per batch passed the
constraint check. With the weight concentrated on a few dozen candidates, the
weighted mean is a noisy proposal, and the step fractions trade progress for
reliability.

The opening decision was also repeated with three optimizer seeds at each of
$128$, $256$, and $512$ candidates. All nine runs lowered the objective of
the supplied schedule, by between $5.3\%$ and $6.7\%$. The effective sample
size grew with the number of candidates, from $4.3$–$18.9$ at $128$ to
$32.5$–$63.0$ at $512$. The final objectives did not improve with the count:
their spread across seeds at a fixed count exceeded the differences between
the averages at different counts. These repetitions measure optimizer
variability at one physical state. They do not establish that the recorded
plan is globally optimal.

Planning took $13.9$ s per decision on average, including the finer checks
and first-use compilation. This is longer than the $2$ s control interval:
the recording is an offline computation, not a demonstration of real-time
control. The numerical model was checked separately by repeating the executed
MPPI controls with halved integration steps, which changes the temperature
field by $0.0014$ K root-mean-square, and on a $128\times64$ mesh, which
changes it by $0.78$ K. The finer mesh reaches $1068.42$ K and a lowest
outlet conversion of $95.36\%$, still within both limits. First-order upwind
advection adds numerical diffusion of roughly
$v_x\Delta x/2\approx0.013$ m$^2$/s, much larger than $D$, so part of the
visible spreading of the patch is a property of the discretization.

### Planning with a learned surrogate

Each recovery decision simulated $256$ candidates over $20$ intervals in each
of four updates, or $20{,}480$ interval rollouts before validation. For this
coarse PDE the budget is affordable. A finer mesh, separate gas and solid
phases, or a computational fluid dynamics model of an industrial calciner
would make it expensive. A common response is to learn a fast approximate
transition model, a surrogate, from simulations and to use it for the
candidate rollouts. MPPI requires only rollouts and costs, so the
substitution changes no part of the update. It changes which model ranks the
candidates: the weights now favor candidates that the surrogate predicts to
be good.

The surrogate experiment uses the uniform-flow version of the reactor with a
different task. Nominal heating is $0.48$ in all four quadrants, and the
inlet temperature $800+10\cos(\pi y/2)+g_t$ K fluctuates. The deviation $g_t$
follows the Ornstein–Uhlenbeck model of the aircraft example with correlation
time $20$ s and stationary standard deviation $15$ K, held constant over each
$2$ s interval and observed at each decision. The objective is economic: use
as little heat as possible while keeping outlet conversion at least $95\%$ and
temperature at most $1070$ K. The planning cost averages, over $20$ steps,

$$
\ell_k=\frac14\sum_{j=1}^4u_{k,j}
+5\left[\frac{(0.95-a_{\mathrm{out},k+1})_+}{0.05}\right]^2
+10\left[\frac{(\max T_{k+1}-1070\ \mathrm K)_+}{20\ \mathrm K}\right]^2
+\frac{0.05}{4}\sum_{j=1}^4\left(\frac{u_{k,j}-u_{k-1,j}}{0.1}\right)^2,
$$

where $(\cdot)_+=\max(\cdot,0)$ and the conversion and temperature are
evaluated at the end of interval $k$.

The surrogate is a residual convolutional network that predicts the change
of both fields over one $2$ s interval. Its six input channels are the
normalized temperature and reactant fields, the map of heater fractions, the
normalized inlet temperature, and two coordinate channels. Four $3\times3$
convolutions with $16$ hidden channels and dilations $1,2,4,1$ let each output
cell depend on inputs up to eight cells away, more than the $3.2$ cells that
material travels in one interval. After each predicted step, the reactant
fraction is projected onto $[0,1]$; temperature is not modified. Training
used $256$ PDE episodes of $120$ s with varied heating, warm and cold
starts, smooth spatial transients, and inlet fluctuations, split by whole
episode into $179$ training, $38$ validation, and $39$ test episodes. The loss
combined one-step error, five-step rollout error, and outlet error. The
checkpoint used here was selected after three epochs of Adam within a local
CPU budget.

| Held-out $40$ s rollouts, $39$ test episodes | Result |
| --- | ---: |
| Temperature root-mean-square error | $4.59$ K |
| Outlet-conversion mean absolute error | $0.0039$ |
| 95th percentile of peak-temperature underprediction | $8.31$ K |
| Largest reactant-fraction bound excess before projection | $0.068$ |

These averages must be read against the margins that decide feasibility.
Under constant heating, the lowest outlet conversion in the trials below is
$95.09\%$, a margin of $0.09$ percentage points, while the surrogate's
average conversion error is $0.39$ percentage points. A plan accepted on the
surrogate's predictions alone could violate the conversion limit in the PDE,
and an $8$ K underprediction of the peak temperature could hide a violation
of the temperature limit.

The controller therefore keeps two roles for the PDE. Every applied control
advances the PDE under the realized inlet, and every proposed plan is
simulated in the PDE before it is accepted. The planner samples $K=128$
candidates in latent coordinates with
$u(z)=\operatorname{sigmoid}(\operatorname{logit}(0.48)+0.35z)$ and the
reference $p_0=\mathcal N(0,0.49I)$, which now decodes to constant nominal
heating. Its KL penalty charges departures from nominal heating, while the
proposal mean is initialized from the shifted previous plan. Candidate costs
come from surrogate rollouts, and the weights use $\lambda=0.025$. A
candidate that uses one percentage point less heat on average, with no
penalty active, receives about $e^{0.01/0.025}\approx1.5$ times the weight.
After each of two updates per decision, the weighted mean is accepted only if
it lowers the surrogate cost and a PDE rollout under the conditional mean
inlet keeps conversion at or above $95\%$ and temperature at or below
$1070$ K at every $2$ s output. Otherwise the previous plan is kept. If the
shifted previous plan fails the PDE check at the next decision, constant
nominal heating is checked instead; if both fail, the episode stops and
records a planning failure.

Two forecast variants separate reaction to the observed inlet from planning
under its uncertainty, as in the aircraft example. The mean-forecast
controller scores each candidate under the conditional mean of the future
inlet. The stochastic controller scores it under four conditional futures,
shared by all candidates, and averages the four costs before weighting.

### Surrogate and PDE planning compared

Eight paired trials start from the warm state and last $160$ s. Within a
trial, constant heating and both MPPI controllers face the same realized
inlet sequence.

| Controller | Mean heating energy | Lowest outlet conversion | Highest temperature | Accepted updates | Planning time per decision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant heating | $76.800$ s | $95.09\%$ | $1059.5$ K | none | none |
| Surrogate MPPI, mean forecast | $75.807$ s | $95.30\%$ | $1051.9$ K | $127$ of $1280$ | $1.32$ s |
| Surrogate MPPI, stochastic forecast | $75.759$ s | $95.16\%$ | $1052.4$ K | $143$ of $1280$ | $4.45$ s |

Lowest and highest values are taken over all trials at the $2$ s outputs. No
episode stopped, and every execution stayed within both limits on the $2$ s
grid and on a separate $0.1$ s audit of the executed controls. Paired
differences in heating energy, with $95\%$ Student-t intervals over the eight
trials, are $-0.99\pm0.30$ s for mean-forecast MPPI minus constant heating,
$-1.04\pm0.34$ s for stochastic MPPI minus constant heating, and
$-0.048\pm0.099$ s for stochastic minus mean-forecast MPPI. Both MPPI
variants use about $1.3\%$ less heat than constant heating. The trials do
not establish a difference between the two forecast variants. About one
proposed update in ten was accepted.

:::{figure} _static/thermal_reactor/paired-trials.svg
:label: fig-reactor-surrogate-trials
:width: 100%
:alt: Three dot plots compare constant heating, mean-forecast MPPI, and stochastic MPPI over eight trials. Heating energy is 76.8 seconds for constant heating and between about 75.1 and 76.4 seconds for both MPPI variants. Lowest outlet conversion stays above the 0.95 line for all trials. Highest temperature stays below the 1070 kelvin line, lower for MPPI than for constant heating.

Eight paired $160$ s trials of surrogate MPPI in the uniform-flow reactor.
Left: heating energy. Center: lowest outlet conversion, with the $95\%$
limit dashed. Right: highest temperature, with the $1070$ K limit dashed.
Each dot is one trial, and the three controllers in a trial share one
realized inlet sequence.
:::

To isolate the effect of the rollout model, the same planner was run from the
warm state with zero inlet deviation, once with surrogate rollouts and once
with PDE rollouts, using the same random seed and settings. Each resulting
$40$ s plan was then evaluated in the PDE on $32$ fresh inlet futures. The
first heater commands were nearly identical,
$(0.484,0.474,0.486,0.480)$ from the surrogate and
$(0.484,0.472,0.486,0.478)$ from the PDE, and the held-out PDE costs were
$0.47331$ and $0.47250$. Planning took $14.57$ s with the surrogate and
$11.25$ s with the PDE, both including PDE validation and the recording of
candidate snapshots. For this coarse PDE the surrogate brought no speed
advantage. A surrogate pays for itself only when the model it replaces is
substantially more expensive to roll out.

The two plans nearly coincide because of the structure of $\ell_k$. For a
candidate that triggers neither penalty, the cost consists of mean heating
and the slew term, which depend only on the controls. The surrogate and the
PDE assign such a candidate the same cost. For both selected plans, the cost
on the planning forecasts agreed with the held-out PDE cost to within
$10^{-6}$. Such agreement between different models and inlet futures means
that the penalties contributed essentially nothing in either evaluation. The
rollout model affects the weights only through the penalties, that is,
through deciding which low-heating candidates violate the conversion limit
and which high-heating candidates exceed the temperature limit. Surrogate
accuracy matters near the boundary of the feasible set, which is also where
an economic objective drives the plan. Average errors over held-out
trajectories do not measure accuracy there. The PDE acceptance check instead
tests the one plan that will be executed, under the conditional mean inlet.

Both experiments use the same structure. Sampling explores a small, bounded
parameterization of the heater schedule; exponential weights with a
likelihood correction propose an update; and the full model decides whether
the update replaces a plan that is already known to be feasible. In the
recovery experiment the full model also generates the candidate costs. In
the surrogate experiment a learned model generates them, and its errors
matter where the constraints are nearly active.

:::{iframe} ../interactive/thermal-reactor-mppi.html#recovery
:width: 100%
:title: Thermal-reactor recovery and surrogate replay
:class: thermal-reactor-replay

The replay opens on the field-recovery experiment. Play the recording to
follow the cold patch under unchanged heating and MPPI, display the error
relative to the operating state, or open the sampled control update to see
six candidates and their full-batch weights at $14$, $22$, and $34$ s. The
Surrogate vs PDE tab compares the learned model with the PDE under identical
controls and inlet histories in the uniform-flow experiment. Every field
image is a recorded numerical rollout; the page performs no simulation.
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

The reactor application uses the same update on a discretized PDE with
4,096 state values. Six bounded knots per heater reduce the search to 24
latent parameters, in which the Gaussian densities and the likelihood
correction are computed, and candidates that violate a limit receive zero
weight. Because the mean of feasible candidates need not be feasible, each
proposed update is checked in the PDE, damped when necessary, and otherwise
rejected; once the field was recovering, most updates were rejected.
Replacing the PDE by a learned surrogate in the sampling loop changes the
model that ranks candidates, while the PDE still decides which plan is
executed. With an economic objective, the surrogate's errors matter mainly
near the operating limits, which is where the plan operates.

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

The aircraft and reactor experiments use small numerical modules and read
precomputed results when the book is built. The aircraft evaluator averages
each candidate's costs over a shared ensemble of possible wind futures.

- {download}`Gaussian importance weights and mean updates <code/mppi_control.py>`
- {download}`Aircraft model, wind scenarios, and trajectory optimizer <code/aircraft_mppi.py>`
- {download}`Aircraft experiment and figure generator <scripts/build_aircraft_mppi_artifacts.py>`
- {download}`ERA5 numerical snapshot <data/aircraft/era5_wind.npz>` and
  {download}`provenance and checksums <data/aircraft/era5_wind.json>`
- {download}`Reactor PDE, heaters, and inlet disturbances <code/thermal_reactor.py>`
  and {download}`shear flow and cold inlet pulse <code/thermal_plume.py>`
- {download}`Direct-PDE field recovery <code/thermal_recovery.py>`
- {download}`Convolutional surrogate <code/thermal_surrogate.py>` and
  {download}`surrogate MPPI with PDE acceptance checks <code/thermal_control.py>`
- {download}`Reactor experiment stages <scripts/build_thermal_reactor.py>`
- {download}`Recovery decisions <artifacts/thermal_recovery/decisions.json>`
  and {download}`surrogate trials <artifacts/thermal_reactor/evaluation.json>`

The experiment scripts record the physical parameters, planner settings,
seeds, and validation diagnostics alongside their results. The
[aircraft replay](interactive/aircraft-mppi.html) and the
[reactor replay](interactive/thermal-reactor-mppi.html) display recorded
simulations without a Python kernel. Regeneration commands are in the repository README.

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

:::{exercise} The computational budget of a reactor decision
:label: ex-mppi-reactor-budget

The field-recovery controller of the [reactor example](#sec-reactor-mppi)
draws $K=256$ latent arrays in each of four updates per decision. Each
candidate is simulated for $20$ intervals of $2$ s, with PDE substeps of at
most $0.1$ s.

**(a)** How many state values does a candidate rollout advance at each
substep, and how many numbers does the weighted mean average?

**(b)** Ignoring validation, how many $2$ s interval simulations and at
least how many substeps does one decision require?

**(c)** The mesh is refined from $64\times32$ to $128\times64$ cells.
Using the advective condition $v_x\Delta t\leq\Delta x$ with
$v_x\leq0.22$ m/s, check that the $0.1$ s substep cap still determines the
substep size. By what factor do the state dimension, the number of sampled
parameters, and the work per decision change?

**(d)** The six-knot parameterization cannot represent every sequence of
$80$ heater values. Give an example of a schedule it excludes, and explain
what this restriction implies for the claim that the recorded MPPI schedule
is optimal.
:::

:::{solution} ex-mppi-reactor-budget
:class: dropdown

**(a)** Each of the $2{,}048$ cells carries a temperature and a reactant
fraction, so a rollout advances $4{,}096$ state values. The weighted mean
averages the $6\times4=24$ latent parameters.

**(b)** One decision simulates $256\times4\times20=20{,}480$ intervals.
Each interval needs at least $2/0.1=20$ substeps, so at least $409{,}600$
substeps are required, each evaluating the right-hand side twice in the
two-stage Runge–Kutta method.

**(c)** The refined mesh has $\Delta x=8/128=0.0625$ m, so the advective
condition allows $\Delta t\leq0.0625/0.22\approx0.28$ s, and the $0.1$ s cap
remains binding. The state dimension grows by a factor of four, to
$16{,}384$. The number of sampled parameters stays $24$. Each substep
updates four times as many cells, so the work per decision grows by about a
factor of four. The weights and the update keep the same dimension.

**(d)** A single $2$ s increase of one heater in the middle of the horizon,
with all neighboring values at nominal, is not piecewise linear between the
six knot times and cannot be represented. MPPI searches only within the
knot family, so the recorded schedule is at best a good member of that
family. Finite sampling, the Gaussian mean fit, the regularizing reference,
and the step fractions limit even that statement, so the result supports an
improvement over the supplied schedule, not optimality.
:::

:::{exercise} Sampling bounded heater inputs
:label: ex-mppi-reactor-latent

A heater knot is decoded by
$u(z)=\operatorname{sigmoid}(\operatorname{logit}(u_{\mathrm{seed}})+sz)$
with $s>0$. The reference is $\mathcal N(0,\sigma^2I)$ and the proposal is
$\mathcal N(\mu,\sigma^2I)$, both over the latent array $z$.

**(a)** Take one scalar knot with $u_{\mathrm{seed}}=0.9$ and $s=0.35$, and
two candidates $z=2$ and $z=-2$ with equal weights. Compare the decoded
weighted mean $u(\bar z)$ with the average of the two decoded inputs. Which
does the reactor controller propose, and why does neither leave $[0,1]$?

**(b)** Show that the weights
$w_i\propto e^{-J_i/\lambda}p_0(z^{(i)})/q_\mu(z^{(i)})$ equal
$w_i\propto e^{-\widetilde J_i/\lambda}$ with corrected cost
$\widetilde J_i=J_i+\lambda\mu^\top z^{(i)}/\sigma^2$, up to a factor common
to all candidates. Compare with the
[corrected cost](#sec-mppi-importance) derived earlier in the chapter.

**(c)** In the recorded first decision, $\lambda=0.01$ and $\sigma^2=0.49$.
In the same batch, candidate 12 had cost $0.05649$ and weight $0.0682$,
while candidate 180 had cost $0.05840$ and weight $0.1565$. Use part (b) to
compute $\mu^\top(z^{(12)}-z^{(180)})$ and interpret its sign.

**(d)** Another implementation samples physical knot values from a Gaussian
and clips them to $[0,1]$ before simulation. How should it compute the
likelihood correction, and what is one disadvantage relative to the
logistic decoder?
:::

:::{solution} ex-mppi-reactor-latent
:class: dropdown

**(a)** Here $\operatorname{logit}(0.9)=\log9\approx2.197$ and $sz=\pm0.7$.
The decoded inputs are $\operatorname{sigmoid}(2.897)\approx0.948$ and
$\operatorname{sigmoid}(1.497)\approx0.817$, whose average is about
$0.882$. The latent mean is $\bar z=0$, which decodes to $0.9$. The
controller averages latent parameters and proposes $u(\bar z)=0.9$, subject
to validation. Both numbers lie in $(0,1)$: the sigmoid maps every real
number into $(0,1)$, and an average of numbers in $(0,1)$ stays there. They
differ because the sigmoid is curved: near the upper bound, equal latent
steps produce a smaller increase than decrease.

**(b)** For Gaussians with a common covariance,

$$
\log\frac{p_0(z)}{q_\mu(z)}
=-\frac{\|z\|^2}{2\sigma^2}+\frac{\|z-\mu\|^2}{2\sigma^2}
=-\frac{\mu^\top z}{\sigma^2}+\frac{\|\mu\|^2}{2\sigma^2}.
$$

The last term is the same for every candidate and cancels after
normalization. Thus
$e^{-J_i/\lambda}p_0/q_\mu\propto\exp[-(J_i+\lambda\mu^\top z^{(i)}/\sigma^2)/\lambda]$.
Writing $z^{(i)}=\mu+\boldsymbol\epsilon^{(i)}$ changes the correction only
by another common constant, giving the form
$\lambda\mu^\top\Sigma^{-1}\boldsymbol\epsilon^{(i)}$ of the earlier corrected
cost, with $\Sigma=\sigma^2I$.

**(c)** The weight ratio gives the difference in corrected costs,

$$
\widetilde J_{12}-\widetilde J_{180}
=-\lambda\log\frac{0.0682}{0.1565}\approx0.0083.
$$

The cost difference is $J_{12}-J_{180}=-0.0019$, so the correction
contributes $0.0102=\lambda\mu^\top(z^{(12)}-z^{(180)})/\sigma^2$. Hence
$\mu^\top(z^{(12)}-z^{(180)})\approx0.0102\times0.49/0.01\approx0.50$. The
positive sign means that candidate 12 lies farther along the direction in
which the proposal mean has already moved. The proposal draws such
candidates more often than the reference would, and the correction lowers
their weight to compensate.

**(d)** The Gaussian density belongs to the unclipped sample, so the ratio
must be computed for that variable, with clipping treated as part of the
deterministic map from samples to controls, as for the aircraft duration.
Every sample beyond a bound then produces the same control and the same
cost. Those samples are indistinguishable to the weights, and the weighted
mean of unclipped values can itself lie outside $[0,1]$. The logistic decoder
keeps distinct latent samples distinct and always decodes the mean inside the
bounds.
:::

:::{exercise} When the rollout model matters
:label: ex-mppi-reactor-surrogate

In the surrogate experiment, each step of a candidate's planning cost is
$\ell_k$, combining mean heating, a conversion penalty, a temperature
penalty, and a slew penalty.

**(a)** Suppose that, for some candidate, neither penalty is active under
the surrogate or under the PDE. Show that both models assign it the same
planning cost. Which candidates can the surrogate's errors affect?

**(b)** The surrogate's 95th-percentile peak-temperature underprediction on
held-out rollouts is $8.31$ K. A proposed plan has a predicted maximum
temperature of $1064$ K. Does this establish that it satisfies the $1070$ K
limit in the PDE? Which step of the controller protects the plant, and
under which inlet sequence does that step check the plan?

**(c)** An alternative tightens the limits used inside surrogate rollouts to
$1070-\delta$ K and $95\%+\varepsilon$. Describe the tradeoff in choosing
$\delta$ and $\varepsilon$ when the objective is to minimize heating.

**(d)** At one state, planning with the surrogate took $14.57$ s and
planning with the PDE took $11.25$ s. When would you expect a surrogate to
reduce planning time? What evidence would you require before removing the
PDE acceptance check?
:::

:::{solution} ex-mppi-reactor-surrogate
:class: dropdown

**(a)** Without active penalties, $\ell_k$ reduces to mean heating plus the
slew term, both functions of the controls alone. Any rollouts that remain
valid therefore give the same cost under either model. Surrogate errors can
change the cost only of candidates for which at least one model predicts an
active penalty, that is, candidates near the conversion or temperature
limit. Minimizing heating pushes the plan toward the conversion limit, so
these are the candidates that matter.

**(b)** No. The $8.31$ K figure is a percentile over held-out windows, not a
bound: in one window in twenty the surrogate underpredicted by more, and an optimized plan is
not drawn from the held-out distribution. Adding $8.31$ K to $1064$ K already
exceeds $1070$ K. The PDE acceptance check simulates the proposed plan before
it replaces the stored one. It uses the conditional mean of the future
inlet, so it does not guarantee feasibility under the inlet that is actually
realized. The audit on a $0.1$ s grid checks the executed controls only after
the fact.

**(c)** Larger margins make surrogate rollouts reject more candidates near
the limits. Fewer proposals then fail the PDE check, and violations under
the realized inlet become less likely, but the controller gives up heating
reductions that were feasible. Smaller margins allow plans closer to the
limits, at the price of more rejected proposals and less protection against
surrogate error and inlet uncertainty. Margins calibrated from surrogate
errors near the limits, rather than from average errors, would make this
choice less arbitrary.

**(d)** A surrogate reduces planning time when rollouts of the full model
dominate the computation and the surrogate is much cheaper to evaluate in
large batches, for example with a finer mesh, multiphase physics, or
computational fluid dynamics. Removing the PDE check would require evidence
about errors near the constraint boundary for the kinds of plans the
optimizer proposes, not only average errors on training-like trajectories,
together with closed-loop trials whose executed controls are audited in the
full model.
:::

:::{exercise} A surrogate for field recovery
:label: ex-mppi-reactor-project

The field-recovery experiment plans with the PDE, while the available
surrogate was trained for the uniform-flow reactor with random inlet
fluctuations. This project closes that gap using the modules listed under
Computational Sources.

**(a)** Generate PDE training episodes under the shear flow that include
cold inlet pulses with varied position, width, depth, and duration, together
with varied heating schedules. Train the residual network and report
held-out errors for the field error $E$, the peak temperature, and the outlet
conversion.

**(b)** Replace PDE rollouts by surrogate rollouts when scoring candidates
in the recovery controller, keeping the PDE validation unchanged. Compare
integrated field error, accepted updates, and planning time with the
direct-PDE recording.

**(c)** Test the controller on a pulse whose position lies outside the
training range. Which recorded diagnostics would show that the surrogate is
unreliable before a plan is executed?
:::

:::{solution} ex-mppi-reactor-project
:class: dropdown

Answers depend on the training data and budget. Useful comparisons keep the
initial state, supplied schedule, planner settings, and optimizer seeds
fixed, so that differences come from the rollout model. Since the tracking
objective depends on the entire field, surrogate errors affect every
candidate's cost, unlike in the economic objective of
Exercise @ex-mppi-reactor-surrogate. When the surrogate ranks candidates
differently from the PDE, an accepted update can lower the surrogate
objective while raising the PDE objective, so both should be recorded for
every proposal. For part (c), the
fraction of proposals rejected by the PDE check, the gap between the
surrogate's predicted objective and the PDE objective of each proposal, and
raw reactant-fraction values outside $[0,1]$ before projection all signal
that the surrogate is operating outside its training distribution.
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

:::{exercise} Randomness in the reactor recovery
:label: ex-mppi-check-reactor

In the field-recovery experiment, which quantities are random: the plant,
the inlet, or the candidate controls? What changes when the optimizer seed
changes?
:::

:::{solution} ex-mppi-check-reactor
:class: dropdown

Only the candidate controls are random. The PDE is deterministic and the
inlet is known after the pulse, so random numbers serve only to explore
heater schedules. A different seed changes the sampled candidates and
therefore the plan. Across the nine repetitions of the opening decision, the
final objective varied by about $1.5\%$.
:::
