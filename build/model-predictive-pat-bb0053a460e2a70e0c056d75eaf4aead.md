# Model Predictive Path Integral Control

Shooting and collocation optimize a trajectory from a specified initial
state. How should a controller plan when physical disturbances make several
future trajectories possible? Model predictive path integral control (MPPI)
uses sampled trajectories to improve a control sequence, and observations
allow that sequence to be revised during execution. A Brownian particle
provides an analytic check, competing passages show uncertainty changing an
action, and an aircraft under uncertain winds extends the sampling method to
a more general simulator.

## Learning Goals

After working through the examples and exercises, you should be able to:

- Calculate an MPPI update, including the correction for sampling around a
  nonzero nominal control sequence.
- Derive the path-integral representation under its noise and control-cost
  assumptions, with consistent continuous- and discrete-time noise scaling.
- Distinguish physical uncertainty from numerical sampling, and identify
  which approximations enter a practical stochastic trajectory optimizer.
- Compare controllers using paired disturbances, numerical reference
  solutions, and explicit feasibility and sampling diagnostics.
- Execute the first action of a sampled plan and update the remaining plan
  from new observations.

## Prerequisites

The algorithm uses the [single-shooting rollout](numerical-trajectory-optimization.md)
and the [finite-horizon optimal-control formulation](discrete-time-optimal-control.md).
Gaussian expectations and Brownian increment scaling are introduced in
[Stochastic Dynamics and Partial Observation](stochastic-dynamics-observation.md).
KL divergence, importance sampling, and the continuous-time value recursion
are developed here; the later dynamic-programming chapters are not prerequisites.

(sec-mppi)=
## Improving a Control Sequence with Sampled Trajectories

How can a control sequence be improved using only forward simulations,
and what objective does a weighted average of sampled inputs solve?

Single shooting evaluates a control sequence by simulating its trajectory.
That same forward computation can compare many randomly perturbed sequences,
even when derivatives of the simulator or the cost are unavailable. Model
predictive path integral control (MPPI) combines these simulations through a
weighted average: perturbations associated with lower costs receive greater
weight. Its computation uses Gaussian sampling, expectations, and the
single-shooting rollout introduced in
[Numerical Trajectory Optimization](numerical-trajectory-optimization.md). Its connection to
stochastic optimal control requires an additional relationship between the
control cost and the process noise.

### A Brownian particle

A microscopic particle suspended in a fluid moves under both an applied
force and thermal fluctuations. In normalized units, a scalar model is

$$
dX_s=u_s\,ds+\sqrt{\nu}\,dW_s,
\qquad \nu>0.
$$

Here $u_s$ is the drift produced by the applied force, and $W_s$ is standard
Brownian motion: an increment over an interval of length $h$ has mean zero
and variance $h$, independently of earlier increments. The particle remains
stochastic when the force is fixed. The objective is to approach a target
while limiting the applied force.

For the sampling calculations, index $N$ transitions by $k=0,\ldots,N-1$,
with states $x_0,\ldots,x_N$ and step length $h$. A constant control during
each step gives the exact scalar transition

$$
x_{k+1}=x_k+h(u_k+\epsilon_k),
\qquad
\epsilon_k\sim\mathcal N(0,\nu/h).
$$

Multiplication by $h$ gives a random position increment with variance
$\nu h$. The control-like perturbation $\epsilon_k$ therefore has a
different variance from the position increment it produces.

Consider one step with $x_0=0$, $h=\nu=1$, nominal control $u_0=0$, and
terminal cost $c_1(x)=(\log 2)(x-1)^2$. Suppose three sampled perturbations,
rounded for the calculation, are $-1,0,1$. Set the weighting parameter to
$\lambda=1$ and assign each trajectory a weight proportional to
$\exp(-c_1(x_1)/\lambda)$:

| Perturbation $\epsilon_0$ | Endpoint $x_1$ | Cost $c_1(x_1)$ | Unnormalized weight | Normalized weight |
| --- | --- | --- | --- | --- |
| $-1$ | $-1$ | $4\log 2$ | $1/16$ | $1/25$ |
| $0$ | $0$ | $\log 2$ | $1/2$ | $8/25$ |
| $1$ | $1$ | $0$ | $1$ | $16/25$ |

The weighted perturbation changes the nominal control to

$$
u_0^+=0+\frac1{25}(-1)+\frac8{25}(0)+\frac{16}{25}(1)
=\frac35.
$$

The update moves toward the target while retaining contributions from all
three trajectories. It is a finite-sample calculation, so $3/5$ is not an
exact optimum. A probability distribution over trajectories will specify
the expectation this weighted average estimates.

### A distribution over control sequences

For a general rollout, write
$\mathbf{x}_{k+1}=\mathbf{f}_k(\mathbf{x}_k,\mathbf{v}_k)$ and collect the
realized inputs into
$\mathbf{V}=(\mathbf{v}_0,\ldots,\mathbf{v}_{N-1})$. The initial state is
fixed. Define its state and terminal cost by

$$
S(\mathbf V)=c_N(\mathbf{x}_N)
+\sum_{k=0}^{N-1}c_k(\mathbf{x}_k).
$$

The absence of a control-effort term in $S$ is deliberate: the distribution
over inputs will supply that penalty. Assume $S$ is measurable and bounded
below. Choose a Gaussian reference distribution $P_0$ and a Gaussian
sampling distribution $Q_{\mathbf U}$ with the same positive-definite
covariance $\Sigma$:

$$
p_0(\mathbf V)=\prod_{k=0}^{N-1}\mathcal N(\mathbf v_k;\mathbf0,\Sigma),
\qquad
q_{\mathbf U}(\mathbf V)=
\prod_{k=0}^{N-1}\mathcal N(\mathbf v_k;\mathbf u_k,\Sigma).
$$

The reference favors small inputs. The sampling mean
$\mathbf U=(\mathbf u_0,\ldots,\mathbf u_{N-1})$ locates the rollouts near
a current candidate. For the Brownian model, these input distributions
describe the drift-plus-noise terms in the time-discrete dynamics. They
also define a sampling optimizer for a deterministic simulator; that use
alone makes no assertion about physical process noise.

To favor low-cost trajectories without discarding the reference, multiply
its density by the cost weight and normalize:

$$
p^\star(\mathbf V)=\frac{e^{-S(\mathbf V)/\lambda}p_0(\mathbf V)}{Z},
\qquad
Z=\mathbb E_{P_0}[e^{-S(\mathbf V)/\lambda}],
\qquad \lambda>0.
$$

Assume $0<Z<\infty$. This is a Gibbs distribution: two trajectories with
equal reference density and cost difference $\Delta S$ have density ratio
$e^{-\Delta S/\lambda}$. The parameter $\lambda$ has the units of cost
and controls how strongly those costs affect the ratio.

The optimization problem solved by this density follows by expanding a
Kullback--Leibler divergence. For densities $q,p$, define
$D_{\mathrm{KL}}(Q\|P)=\mathbb E_Q[\log(q/p)]$, with $Q$ absolutely
continuous with respect to $P$. Substituting $p^\star$ gives

$$
\begin{aligned}
D_{\mathrm{KL}}(Q\|P^\star)
&=\mathbb E_Q\left[
\log\frac{q(\mathbf V)}{p_0(\mathbf V)}
+\frac{S(\mathbf V)}\lambda+\log Z\right],\\
\mathbb E_Q[S]+\lambda D_{\mathrm{KL}}(Q\|P_0)
&=-\lambda\log Z+\lambda D_{\mathrm{KL}}(Q\|P^\star).
\end{aligned}
$$

The rightmost divergence is nonnegative and vanishes at $Q=P^\star$.
Thus the Gibbs law minimizes expected trajectory cost plus a penalty for
departing from the reference, over distributions for which these
quantities are finite. For the Gaussian family,

$$
\lambda D_{\mathrm{KL}}(Q_{\mathbf U}\|P_0)
=\frac\lambda2\sum_{k=0}^{N-1}
\mathbf u_k^\top\Sigma^{-1}\mathbf u_k.
$$

The distribution penalty becomes quadratic control effort. Adding that
same effort penalty to $S$ would define a different objective and count
the intended penalty twice.

### Gaussian fitting and importance sampling

The Gibbs density is generally neither Gaussian nor independent across
time. A Gaussian mean can nevertheless approximate it by minimizing
$D_{\mathrm{KL}}(P^\star\|Q_{\mathbf M})$ over the candidate mean
$\mathbf M$. The part depending on $\mathbf M$ is

$$
\frac12\mathbb E_{P^\star}
\left[\sum_{k=0}^{N-1}
(\mathbf v_k-\mathbf m_k)^\top\Sigma^{-1}
(\mathbf v_k-\mathbf m_k)\right].
$$

Differentiation with respect to $\mathbf m_k$ gives
$\Sigma^{-1}(\mathbf m_k-\mathbb E_{P^\star}\mathbf v_k)$.
Setting this derivative to zero yields
$\mathbf u_k^+=\mathbb E_{P^\star}\mathbf v_k$, assuming finite
second moments. This is an exact Gaussian mean fit. Its KL direction is
reversed relative to the preceding optimization identity, so it need not
minimize the original objective restricted to Gaussian distributions.
This distinction is part of the information-theoretic formulation
{cite:p}`Williams2017InformationTheoretic`.

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

The density ratio corrects the unequal sampling frequencies caused by
moving the proposal mean. Since both Gaussians have full support, every
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

### Receding-horizon execution

An optimized sequence determines a present action and a proposed future.
After applying its first action, a new observation supplies a new initial
condition. MPPI repeats the trajectory computation from that condition,
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

(sec-path-integral-control)=
## Path-Integral Stochastic Control

When does an exponential average over stochastic trajectories compute
an optimal feedback action?

The exponential weighting also follows from a class of continuous-time
stochastic control problems. Its derivation uses the short-time dynamic
programming principle, a second-order expansion for Brownian increments,
and a change of variables in the value function. These steps explain both
the term *path integral* and the restrictions connecting the sampled
algorithm to an optimal feedback law.

### Control and noise matching

Let $\mathbf X_s\in\mathbb R^n$ follow the control-affine diffusion

$$
d\mathbf X_s=
[\mathbf a(s,\mathbf X_s)+D(s,\mathbf X_s)\mathbf u_s]\,ds
+B(s,\mathbf X_s)\,d\mathbf W_s.
$$

The drift $\mathbf a$ describes uncontrolled motion, $D$ maps the
$m$-dimensional control into state motion, and $B$ maps Brownian noise
into state fluctuations. The letters $D$ and $B$ here distinguish these
matrices from the chapter's NLP constraint maps. Controls may depend on
the current state and time, but not on future noise.

For a fixed terminal time $t_f$, define

$$
V(t,\mathbf x)=\inf_{\mathbf u}
\mathbb E_{t,\mathbf x}^{\mathbf u}\left[
c_f(\mathbf X_{t_f})+
\int_t^{t_f}\left(c(s,\mathbf X_s)
+\frac12\mathbf u_s^\top R\mathbf u_s\right)ds\right],
\qquad R\succ0.
$$

The expectation is conditional on $\mathbf X_t=\mathbf x$. Assume the
coefficients are sufficiently regular for the diffusion to exist without
explosion, the costs are bounded below with finite expectations, and the
value function is once continuously differentiable in time and twice in
state. These assumptions permit the following classical differential
calculation. The control set is $\mathbb R^m$, and the noise and effort
must satisfy

$$
B B^\top=\lambda D R^{-1}D^\top,\qquad \lambda>0.
$$

The equality is pointwise in time and state. It matches both the directions
and magnitudes of control and noise. For the scalar Brownian particle,
$D=1$, $B=\sqrt\nu$, and $R=r$, so $\lambda=r\nu$. Independent choices
of all three quantities would generally violate the equality. Unactuated
state coordinates are allowed, but additional independent noise outside
the control directions does not satisfy this model.

### The stochastic Hamilton--Jacobi--Bellman equation

Over a short interval $\delta$, dynamic programming compares the immediate
cost with the remaining optimal cost:

$$
V(t,\mathbf x)=\inf_{\mathbf u}
\mathbb E\left[
\left(c(t,\mathbf x)+\frac12\mathbf u^\top R\mathbf u\right)\delta
+V(t+\delta,\mathbf X_{t+\delta})\right]+o(\delta).
$$

For this expansion the control is fixed over the short interval.
The state increment has mean
$(\mathbf a+D\mathbf u)\delta+o(\delta)$ and covariance
$BB^\top\delta+o(\delta)$. Its second-order contribution therefore
survives division by $\delta$. Expanding the last expectation gives

$$
\mathbb E[V(t+\delta,\mathbf X_{t+\delta})]
=V+\delta\left[
V_t+(\mathbf a+D\mathbf u)^\top\nabla V
+\frac12\operatorname{tr}(BB^\top\nabla^2V)\right]+o(\delta).
$$

Subtracting $V$ and taking $\delta$ to zero gives the stochastic
Hamilton--Jacobi--Bellman (HJB) equation. The minimizing control solves
a quadratic problem:

$$
R\mathbf u+D^\top\nabla V=\mathbf0,
\qquad \mathbf u^\star=-R^{-1}D^\top\nabla V.
$$

Substitution removes the minimization and leaves

$$
0=V_t+c+\mathbf a^\top\nabla V
-\frac12\nabla V^\top D R^{-1}D^\top\nabla V
+\frac12\operatorname{tr}(BB^\top\nabla^2V),
\qquad V(t_f,\mathbf x)=c_f(\mathbf x).
$$

The term quadratic in $\nabla V$ makes this equation nonlinear, even
though the control minimization itself was explicit.

### The logarithmic transformation

Define a positive function $\Psi$ by $V=-\lambda\log\Psi$. Its derivatives
separate into terms linear and quadratic in derivatives of $\Psi$:

$$
V_t=-\lambda\frac{\Psi_t}{\Psi},\qquad
\nabla V=-\lambda\frac{\nabla\Psi}{\Psi},\qquad
\nabla^2V=-\lambda\frac{\nabla^2\Psi}{\Psi}
+\lambda\frac{\nabla\Psi\nabla\Psi^\top}{\Psi^2}.
$$

After substitution into HJB, the quadratic terms combine as

$$
\frac{\lambda}{2\Psi^2}
\nabla\Psi^\top
\left(BB^\top-\lambda DR^{-1}D^\top\right)\nabla\Psi.
$$

Noise matching sets this expression to zero. Multiplying the remaining
equation by $-\Psi/\lambda$ produces the linear backward equation

$$
\Psi_t+\mathbf a^\top\nabla\Psi
+\frac12\operatorname{tr}(BB^\top\nabla^2\Psi)
-\frac{c}{\lambda}\Psi=0,
\qquad
\Psi(t_f,\mathbf x)=e^{-c_f(\mathbf x)/\lambda}.
$$

This cancellation is the restriction behind the path-integral solution
{cite:p}`Kappen2005PathIntegrals`. An arbitrary control penalty or arbitrary
additional diffusion leaves a nonlinear equation after the transformation.

### A forward expectation

The linear equation has a probabilistic solution. Under the uncontrolled
process, $d\mathbf X_s=\mathbf a\,ds+B\,d\mathbf W_s$, consider

$$
M_s=\exp\left[-\frac1\lambda\int_t^s c(r,\mathbf X_r)\,dr\right]
\Psi(s,\mathbf X_s).
$$

The stochastic chain rule, or Itô's formula, expands changes in $\Psi$
using the same first and second derivatives as the short-time calculation.
The drift of $M_s$ is its exponential factor times
$\Psi_t+\mathbf a^\top\nabla\Psi+
\tfrac12\operatorname{tr}(BB^\top\nabla^2\Psi)-c\Psi/\lambda$,
which vanishes. Assuming the remaining stochastic integral is integrable
and has mean zero, $\mathbb E[M_{t_f}\mid\mathbf X_t=\mathbf x]=M_t$.
Using the terminal condition therefore yields

$$
\Psi(t,\mathbf x)=
\mathbb E_{P_0}\left[
e^{-S/\lambda}\mid\mathbf X_t=\mathbf x\right],
\qquad
S=c_f(\mathbf X_{t_f})+\int_t^{t_f}c(s,\mathbf X_s)\,ds.
$$

This is the Feynman--Kac representation. The expectation integrates over
whole sample paths of the uncontrolled diffusion, hence the name *path
integral*. It replaces a backward partial differential equation with
forward simulations starting at the particular state of interest.
The quadratic control cost is absent from $S$ because its effect is
already incorporated through noise matching and the logarithm.

For the scalar particle with terminal cost
$c_f(x)=\tfrac\kappa2(x-a)^2$ and no running state cost, the uncontrolled
endpoint is $X_{t_f}\sim\mathcal N(x,\nu\tau)$, where
$\tau=t_f-t>0$. The expectation is an ordinary Gaussian integral:

$$
\Psi(t,x)=\frac1{\sqrt{2\pi\nu\tau}}
\int_{\mathbb R}\exp\left[-\frac12\left(
\frac{(y-x)^2}{\nu\tau}+\frac\kappa\lambda(y-a)^2
\right)\right]dy.
$$

Both terms in the exponent are quadratic in the endpoint $y$. Completing
the square with
$\bar y=(\lambda x+\kappa\nu\tau a)/(\lambda+\kappa\nu\tau)$ gives

$$
\frac{(y-x)^2}{\nu\tau}+\frac\kappa\lambda(y-a)^2
=\left(\frac1{\nu\tau}+\frac\kappa\lambda\right)(y-\bar y)^2
+\frac{\kappa(x-a)^2}{\lambda+\kappa\nu\tau}.
$$

The constant term leaves the integral, and integrating the remaining
Gaussian supplies its normalization factor. Using $\lambda=r\nu$ yields

$$
\Psi(t,x)=
\left(1+\frac{\kappa\tau}{r}\right)^{-1/2}
\exp\left[-\frac{\kappa(x-a)^2}
{2\lambda(1+\kappa\tau/r)}\right].
$$

Taking its logarithm and differentiating gives a value and feedback law
that can be checked without Monte Carlo:

$$
V(t,x)=\frac{\kappa r(x-a)^2}{2(r+\kappa\tau)}
+\frac\lambda2\log\left(1+\frac{\kappa\tau}{r}\right),
\qquad
u^\star(t,x)=\frac{\kappa(a-x)}{r+\kappa\tau}.
$$

Noise increases the optimal expected cost through the logarithmic term.
For this quadratic example it does not change the feedback law, a form
of certainty equivalence. Nonquadratic state costs or barriers are needed
to observe decisions that depend on the diffusion strength.

For the opening three-trajectory calculation, $r=\nu=\tau=1$,
$x=0$, $a=1$, and $\kappa=2\log2$. The analytic first action is therefore
$2\log2/(1+2\log2)\approx0.581$, compared with the three-sample update
$0.6$. A second check with $\kappa=4$, $r=1$, $\nu=0.5$, and $\tau=1$
has exact action $0.8$. The saved 200,000-sample calculation estimates
$0.803$ for the first action with the proposal correction; omitting it
instead approaches $0.88$. Sampling error and a missing likelihood ratio
therefore produce different, testable discrepancies.

### Estimating the control from noise increments

The value representation still appears to require a state derivative to
obtain the control. Correlation between the first noise increment and
the future trajectory weight provides that derivative. For an explicit
formula, take $D$ to have full column rank and write $B=DL$, with constant
invertible $L$ satisfying $LL^\top=\lambda R^{-1}$. Conditional on a
short initial increment, the remaining expected weight is
$\Psi(t+\delta,\mathbf X_{t+\delta})$ times the initial running-cost
factor. Its first-order expansion and
$\mathbb E[\Delta\mathbf W\Delta\mathbf W^\top]=\delta I$ give

$$
\mathbb E_{P_0}[e^{-S/\lambda}L\Delta\mathbf W]
=\delta\,LL^\top D(t,\mathbf x)^\top
\nabla\Psi(t,\mathbf x)+o(\delta).
$$

The deterministic terms have zero correlation with the mean-zero noise.
Dividing by $\delta\Psi$ recovers the optimal control:

$$
\mathbf u^\star(t,\mathbf x)=
\lim_{\delta\downarrow0}
\frac{\mathbb E_{P_0}
[e^{-S/\lambda}L(\mathbf W_{t+\delta}-\mathbf W_t)]}
{\delta\,\mathbb E_{P_0}[e^{-S/\lambda}]}.
$$

Low-cost future paths receive greater weight, and their initial random
displacements determine which control direction to apply now. The
formula is conditional on the specified current state; estimates for
future times require the corresponding state conditioning.

### Sampling around a controlled trajectory

Uncontrolled simulations may rarely reach the target. Instead, simulate
under an adapted sampling control $\bar{\mathbf u}$ with the same
diffusion. Girsanov's change-of-measure formula gives

$$
\frac{dP_0}{dQ_{\bar{\mathbf u}}}
=\exp\left[-\frac1\lambda\left(
\frac12\int_t^{t_f}\bar{\mathbf u}_s^\top R\bar{\mathbf u}_s\,ds
+\int_t^{t_f}\bar{\mathbf u}_s^\top RL\,d\mathbf W_s^{\bar{\mathbf u}}
\right)\right].
$$

Here $\mathbf W^{\bar{\mathbf u}}$ is Brownian motion under the sampling
law. Sufficient conditions include the usual exponential-integrability
condition
$\mathbb E_{Q_{\bar{\mathbf u}}}
\exp[\tfrac12\int_t^{t_f}\|L^{-1}\bar{\mathbf u}_s\|^2ds]<\infty$,
which ensures that this likelihood ratio defines a probability measure.
The ratio is the continuous-time counterpart of the Gaussian density
expansion above: the accumulated squared mean shift supplies the first
integral and the mean-noise cross terms supply the second.

Absorb the ratio into a corrected path cost,

$$
S^{\bar{\mathbf u}}=S+
\frac12\int_t^{t_f}\bar{\mathbf u}_s^\top R\bar{\mathbf u}_s\,ds
+\int_t^{t_f}\bar{\mathbf u}_s^\top RL\,d\mathbf W_s^{\bar{\mathbf u}}.
$$

Its stochastic integral has mean zero but must remain inside the
exponential weight. The control estimator then becomes

$$
\mathbf u^\star(t,\mathbf x)=\bar{\mathbf u}(t,\mathbf x)+
\lim_{\delta\downarrow0}
\frac{\mathbb E_{Q_{\bar{\mathbf u}}}
[e^{-S^{\bar{\mathbf u}}/\lambda}L\Delta\mathbf W^{\bar{\mathbf u}}]}
{\delta\,\mathbb E_{Q_{\bar{\mathbf u}}}
[e^{-S^{\bar{\mathbf u}}/\lambda}]}.
$$

The weighted perturbation corrects the current sampling control.
The exact result holds for different admissible sampling controls,
although their Monte Carlo variances can differ substantially
{cite:p}`Thijssen2015Feedback`.

### Discretization and the MPPI update

For simulation with step $h$, Euler--Maruyama gives

$$
\mathbf x_{k+1}=\mathbf x_k+h\mathbf a_k
+hD_k(\mathbf u_k+\boldsymbol\epsilon_k),
\qquad
\boldsymbol\epsilon_k=\frac{L\boldsymbol\xi_k}{\sqrt h},
\qquad \boldsymbol\xi_k\sim\mathcal N(\mathbf0,I).
$$

The covariance required to represent the physical diffusion is therefore
$\Sigma_h=\lambda R^{-1}/h$, so
$\lambda\Sigma_h^{-1}=hR$. Approximating the integrals in
$S^{\bar{\mathbf u}}$ yields

$$
S_h+\sum_{k=0}^{N-1}h\left(
\frac12\mathbf u_k^\top R\mathbf u_k
+\mathbf u_k^\top R\boldsymbol\epsilon_k\right).
$$

For a deterministic nominal sequence, the first term inside the sum is
common to every rollout and cancels in normalized weights. The remaining
cross term is exactly the discrete likelihood correction
$\lambda\sum_k\mathbf u_k^\top\Sigma_h^{-1}\boldsymbol\epsilon_k$.
For a state-dependent sampling controller, the quadratic term generally
varies across trajectories and must be retained.

The discrete and continuous calculations thus produce the same weighted
perturbation expression under matched scaling. Their conclusions have
different scopes. The continuous expression is an optimal current
feedback action in a small-step, exact-expectation limit. The discrete
distributional expression fits an entire Gaussian open-loop mean. Actual
MPPI adds time discretization, finite sampling, and repeated execution of
the first action. None of these steps supplies a general global-optimality
or feasibility guarantee for nonlinear constrained trajectory optimization.

The expectation in the original control objective remains an ordinary
expected cost. Exponential weighting over the uncontrolled diffusion
solves that objective because of the HJB transformation and noise
matching. Exponentiating costs from arbitrary disturbance scenarios of
an unrelated model does not establish the same result. In particular,
physical disturbance uncertainty and extra numerical exploration must
be specified separately whenever the matched diffusion interpretation
does not apply.

(sec-brownian-passages)=
## Brownian Motion Through Competing Passages

When does uncertainty change the direction of the optimal force? The quadratic
example cannot answer this question because its optimal action has certainty
equivalence. We now keep the same fully observed particle and quadratic effort,
and change the geometry of the state cost.

Imagine a particle starting at $x=-0.3$. Between times $1$ and $2$, it should
stay in one of two openings: a narrow interval $[-0.18,0.18]$ near its starting
position, or a wider interval $[-1.7,-0.7]$ farther to the left. One possible
trajectory moves right into the narrow opening, stays there while the passage
is active, and then approaches the terminal target $x=0$ at time $3$. Another
moves left into the wider opening and returns toward the target afterward.
With weak fluctuations, the shorter route requires less effort. Stronger
fluctuations make remaining inside the narrow opening more difficult.

The horizontal coordinate in the figures is **time**, and the vertical
coordinate is the particle's scalar position. The openings describe a cost
that depends on both time and position; they are not a second dynamical state.
This construction follows the double-slit and unequal-passage examples of
{cite:t}`Kappen2005PathIntegrals`. The finite penalties and numerical parameters
below are adaptations for a runnable teaching example.

:::{figure} _static/brownian_mppi/brownian-passages.svg
:label: fig-brownian-passages
:width: 100%

Paired realizations at two physical noise levels. Gray regions incur a finite
penalty. Blue paths use sampled path-integral feedback; dashed orange paths use
feedback obtained by planning under mean dynamics. The same Brownian increments
act on both controllers within each trial. Here $r=1$ and $\lambda=\nu r$.
:::

### A finite-width double slit with an analytic solution

First replace the interval of passage times by one instant $t_s<T$. The particle
pays a lump penalty $F$ if $X_{t_s}$ lies outside a union of intervals
$\mathcal A=\bigcup_j[a_j,b_j]$, and a terminal cost $\kappa X_T^2/2$.
The diffusion and effort remain

$$
dX_s=u_s\,ds+\sqrt\nu\,dW_s,
\qquad \frac r2u_s^2,
\qquad \lambda=\nu r.
$$

The desirability is an expectation over passive Brownian motion. Conditioning
on $Y=X_{t_s}$ separates the crossing from the terminal transport:

$$
\Psi(t,x)=\int_{\mathbb R}
\mathcal N(y;x,\nu(t_s-t))
 e^{-F\mathbf1_{\{y\notin\mathcal A\}}/\lambda}
 \Psi_{\mathrm{free}}(t_s,y)\,dy.
$$

The quadratic calculation above gives

$$
\Psi_{\mathrm{free}}(t,x)=
\sqrt{\frac r{r+\kappa(T-t)}}
\exp\!\left[-\frac{\kappa x^2}{2\nu(r+\kappa(T-t))}\right].
$$

Multiplying the two Gaussian factors and completing the square leaves a
normalized Gaussian in $y$. Set

$$
\delta=t_s-t,\quad A=\frac r\kappa+T-t_s,\quad
\beta=\frac A{A+\delta},\quad m=\beta x,\quad
s^2=\nu\delta\beta.
$$

Its probability of lying in an opening is

$$
P_{\mathcal A}(x)=\sum_j\left[
\Phi\!\left(\frac{b_j-m}{s}\right)
-\Phi\!\left(\frac{a_j-m}{s}\right)\right],
$$

where $\Phi$ is the standard normal cumulative distribution function. Thus

$$
\Psi(t,x)=\Psi_{\mathrm{free}}(t,x)
\underbrace{\left[e^{-F/\lambda}
 +(1-e^{-F/\lambda})P_{\mathcal A}(x)\right]}_{G(t,x)},
\qquad
u^*(t,x)=-\frac{\kappa x}{r+\kappa(T-t)}
+\nu\frac{\partial_xG(t,x)}{G(t,x)}.
$$

This is an analytic benchmark for finite passage widths and a finite penalty.
Taking $F\to\infty$ prohibits crossing outside an opening. For symmetric
openings shrinking to points at $\pm a$, their equal terminal costs cancel
from the control ratio, giving

$$
u^*(t,x)=\frac{a\tanh(ax/(\nu\delta))-x}{\delta}.
$$

At $x=0$, symmetry forces $u^*=0$. The response to a small displacement is

$$
\partial_xu^*(t,0)=\frac1\delta
\left(\frac{a^2}{\nu\delta}-1\right).
$$

If $\nu\delta>a^2$, the force returns a small displacement toward the center:
there is time to postpone commitment. If $\nu\delta<a^2$, the force amplifies
the displacement toward one opening. A realized fluctuation can then determine
which route the feedback follows. At the exact symmetry point, an action of
zero expresses symmetry; it does not guarantee that a single realized path
will pass through the center of either opening.

:::{figure} _static/brownian_mppi/brownian-feedback.svg
:label: fig-brownian-feedback
:width: 100%

Left: the symmetric narrow-slit limit, with $a=\nu=1$, changes its response near
$x=0$ as the crossing approaches. Right: forward importance sampling is checked
against the finite-width formula at $x=0.2$, using openings $[-1.15,-0.85]$ and
$[0.85,1.15]$, $t_s=2$, $T=3$, $r=\kappa=\nu=1$, and $F=20$. Each estimate
averages 32 independent batches of 4096 samples; bars show 95% confidence
intervals for that mean.
:::

### Staying inside unequal passages

A finite-thickness passage requires staying in an opening for a duration.
For the geometry in @fig-brownian-passages, define

$$
q(t,x)=40\,\mathbf1_{\{1<t\leq2\}}
\mathbf1_{\{x\notin[-0.18,0.18]\cup[-1.7,-0.7]\}},
\qquad
J(u)=\mathbb E\!\left[
\frac12X_3^2+\int_0^3\left(q(s,X_s)+\frac12u_s^2\right)ds
\right].
$$

We hold $r=\kappa=1$ fixed and compare $\nu=0.015$ with $\nu=0.3$.
The matching condition requires changing $\lambda$ from $0.015$ to $0.3$ at
the same time. Keeping a separate fixed temperature would change the
relationship to this stochastic-control problem.

The simulation uses $h=0.025$ and evaluates the passage penalty at the right
endpoint of each transition. At each decision, the path-integral controller
samples 4096 futures up to the end of the passage. The terminal quadratic tail
from time $2$ to time $3$ is integrated analytically. Three Gaussian trajectory
proposals generate the futures: one guides the particle toward each opening,
and one retains zero drift. Their mixture improves coverage when successful
passage trajectories are rare under passive sampling.

The trajectory probability of this mixture must appear in the correction.
If $q_j$ is the density of the complete sequence of increments under route
$j$, the denominator is $q_{\mathrm{mix}}=\frac13\sum_{j=1}^3q_j$.
The same route is chosen for an entire simulated future, so a product of
per-step mixture densities would describe a different sampling procedure.
With sampled increments $\Delta x^{(i)}$, the executed action is

$$
w_i\propto e^{-S_i/\lambda}\frac{p_0(\Delta x^{(i)})}
 {q_{\mathrm{mix}}(\Delta x^{(i)})},
\qquad
u(t,x)\approx\frac1h\sum_iw_i\Delta x^{(i)}_0.
$$

The guiding routes choose where to sample. The weighted first increment
chooses the actual force. The physical Brownian increment applied after this
choice comes from a separate random-number stream.

:::{figure} _static/brownian_mppi/brownian-samples.svg
:label: fig-brownian-samples
:width: 100%

Futures from one high-noise update at $t=0$. Blue curves are among the
highest-weight samples; gray curves have smaller weights. The right panel
shows how many samples carry most of the normalized weight. The effective
sample size, $1/\sum_iw_i^2$, diagnoses weight concentration rather than
measuring the number of physically possible futures.
:::

An independent reference evaluates the discretized Feynman--Kac expectation
by backward Gaussian convolution on a position grid:

$$
\Psi_k(x)=\int\mathcal N(y;x,\nu h)
 e^{-h q(t_{k+1},y)/\lambda}\Psi_{k+1}(y)\,dy,
\qquad \Psi_N(x)=e^{-x^2/(2\lambda)}.
$$

Differentiating the Gaussian kernel gives the same first-increment estimator
without Monte Carlo sampling. This is a reference for the time-discretized
path-integral formula. Executing its force as piecewise constant feedback
still introduces a time-step approximation to continuous-time optimal
control. We do not identify its finite-step realized cost with the exact
continuous-time value.

The comparison controller solves a deterministic dynamic program with the
same state cost and control effort, using $x_{k+1}=x_k+hu_k$. It applies its
first action to the observed noisy state and recomputes the action after each
step. Both methods therefore use feedback; they differ in whether planning
accounts for the future diffusion. The deterministic position grid has spacing
$0.0025$, corresponding to action spacing $0.1$, with a numerical action limit
of $24$. The reference convolution uses spacing $0.005$ on $[-6,6]$.

### Paired trials and numerical checks

The experiment uses 128 independent plant disturbance sequences, paired across
controllers. At low noise, sampled path-integral feedback has mean cost
$0.221\pm0.085$, compared with $5.310\pm0.914$ for mean-dynamics planning.
At high noise, the respective costs are $1.808\pm0.289$ and
$11.303\pm1.184$. These are means and normal-approximation 95% confidence
intervals over trials, not bounds on a future trajectory. The numerical
feedback reference gives $0.197\pm0.078$ and $1.767\pm0.279$.

At high noise, the sampled controller occupies the wide passage at time $1.5$
in 126 of 128 trials, compared with 32 of 128 for mean-dynamics planning.
The corresponding Wilson 95% intervals are $[0.945,0.996]$ and
$[0.183,0.332]$. Wide-passage occupancy is measured across all trials,
including trials that incur a penalty. At low noise, neither controller
occupies the wide passage in these trials.

The finite penalty permits failure. We count a trial as a failure if any
checked position lies outside both openings during $1<t\leq2$. Sampled
path-integral feedback fails in 17 of 128 low-noise trials and 28 of 128
high-noise trials; mean-dynamics planning fails in 94 and 122 trials.
At high noise, mean effort is $1.142$ for sampled feedback and $1.531$ for
mean-dynamics planning. The difference in total cost therefore includes both
control effort and time spent outside the openings.

:::{figure} _static/brownian_mppi/brownian-outcomes.svg
:label: fig-brownian-outcomes
:width: 100%

Costs and passage outcomes under paired physical disturbances. Cost bars are
95% intervals for means; proportion bars are Wilson 95% intervals. The numerical
reference and the sampled controller have similar outcomes. The finite
penalty does not impose a probability-of-failure guarantee.
:::

Sampling and time discretization have different effects. At the high-noise
initial state, increasing the candidate count from 256 to 16384 reduces the
first-action root-mean-square error against the convolution reference from
$0.910$ to $0.103$ across 32 independent batches. Refining the time step from
$0.05$ to $0.00625$ changes the reference first action from $-0.681$ to
$-0.749$. Near the route-switching regime, this sensitivity is larger:
for $\nu=0.15$, the same refinement changes the action from $-0.173$ to
$-0.412$. The finite-grid demonstration should therefore be interpreted at its
stated resolution. Crossings between checked times are not counted.

Refining the deterministic grid from $0.0025$ to $0.00125$ changes its paired
mean cost from $5.310$ to $4.915$ at low noise and from $11.303$ to $11.446$
at high noise. No executed control reaches the numerical action limit.
The cost gap persists under this refinement, although the deterministic
baseline itself is not fully converged. Expanding the convolution domain
from $[-6,6]$ to $[-8,8]$ leaves its initial high-noise action unchanged to
the displayed precision. All settings, trial arrays, and convergence sweeps
are retained in the {download}`experiment results <artifacts/brownian_mppi/results.json>`.

:::{iframe} ../interactive/brownian-mppi.html
:width: 100%
:title: Brownian passage replay
:class: mppi-replay

Replay a paired trial, change the physical noise level, and inspect accumulated
control effort and outside visits. The replay uses saved trajectories.
:::

Before moving on, consider why the mean-dynamics controller often waits near
an opening's edge. Which term in its predicted objective assigns a cost to
remaining close to that edge, and which future events does that prediction
omit? The aircraft example retains the distinction between a known present
state and uncertain futures, while using a model outside the matched-noise
class.


(sec-aircraft-mppi)=
## Aircraft Planning Under Uncertain Winds

How can trajectory sampling use a stochastic simulator when the path-integral
matching condition does not hold? An aircraft provides a concrete example:
wind affects its ground motion, while the pilot's commands also affect speed,
altitude, and fuel consumption. The distributional construction of MPPI still
provides a sampling method, but the Brownian stochastic-control theorem no
longer identifies its weighted update with optimal feedback.

Consider an Airbus A320 flying from Montréal–Trudeau (CYUL) to Toronto–Pearson
(CYYZ). The task is to choose a complete airborne trajectory, including climb,
cruise, descent, and flight duration, with low fuel consumption. A favorable
wind at one altitude may justify climbing, but the wind encountered later is
uncertain. A plan computed at departure can therefore miss its intended
arrival even when the aircraft follows every planned command.

### The airborne model and its wind state

The implementation uses the numerical OpenAP A320 model {cite:p}`Sun2022`.
Its state is $(p_E,p_N,h,m,t)$: east and north position in a local map,
altitude in meters, mass in kilograms, and elapsed time in seconds. Controls
are Mach number $M$, vertical speed $v_z$ in meters per second, and heading
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

The heading convention and flight-path-angle approximation reproduce
`OpenAP.top`'s point-mass equations. Conversions to knots, feet, and feet per
minute occur only when calling OpenAP interfaces that require them. The
numerical derivatives are tested against `OpenAP.top`'s original symbolic
model at climb, cruise, and descent states. Optimization itself uses forward
simulation and Gaussian sampling.

The aircraft starts with $85\%$ of maximum takeoff mass and an altitude of
$100$ ft, and aims to arrive at the same altitude. This is a complete airborne
leg, with a nominal cruise altitude of $7000$ m and a nominal duration of
$3600$ s. It omits taxi, runway operations, and traffic. Mach number, vertical
speed, heading, altitude, mass, and the geographic flight corridor have
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

The additional gust $g=(g_E,g_N)$ follows an Ornstein–Uhlenbeck model. Its
exact transition over a time increment $\Delta$ is

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

### Averaging disturbances before weighting candidates

A candidate is a vector $z\in\mathbb R^{13}$: one duration parameter and four
interior knots for each of the three control channels. The knots define small,
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

At a decision time, draw a shared ensemble of $L$ gust futures conditional on
the observed $g_t$. For every candidate $z^{(i)}$, execute the same wind futures
through the aircraft simulator and estimate

$$
\widehat C(z^{(i)})=\frac1L\sum_{\ell=1}^L
 C\bigl(z^{(i)},W^{(\ell)}\bigr).
$$

The cost $C$ is fuel burned, plus quadratic penalties for horizontal and
vertical arrival error and for envelope violations. Arrival error is scaled
by tolerances of $1000$ m horizontally and $30$ m vertically. A squared unit
of either scaled arrival error adds $80$ kg-equivalent cost; envelope
violations receive a larger penalty. The code records the physical errors
separately from this combined objective.

Use a reference $p_0=\mathcal N(0,0.7^2I)$ and proposal
$q_\mu=\mathcal N(\mu,0.7^2I)$ in the **latent knot and duration coordinates**.
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

Averaging must precede exponentiation. For example, a candidate with costs
$0$ and $20$ across two winds has mean cost $10$, whereas one with costs $8$
and $8$ has mean cost $8$. Exponentiating individual costs and then averaging
would favor the first candidate at a small temperature because of its one
fortunate wind realization. That would define a different objective.

The reference distribution regularizes the latent parameters, and the
finite-sample Gaussian projection supplies a practical search direction.
The physical OU gusts are separate from parameter exploration. They neither
act in the same directions as the controls nor satisfy the Brownian
covariance–effort matching relation. Consequently, this disturbance-averaged
construction is a stochastic shooting application of MPPI-style updates,
not the matched-noise estimator derived in @sec-path-integral-control.
This distinction follows the separation of distributional and
stochastic-control derivations in {cite:t}`Williams2017InformationTheoretic`.

### Accepting and executing a plan

Prediction uses 60 s control intervals. Before accepting a weighted mean,
the code constructs its control tape and replays it on a 10 s grid under the
conditional mean wind. It checks the envelope and both arrival tolerances.
The proposed plan must also improve the estimated cost on the shared wind
ensemble. An invalid or non-improving update leaves the validated incumbent
in place. The initial geometric reference must pass the same validation. At a later
decision, the remaining previous plan is also replayed from the observed
state and can be retained as the incumbent. Validation includes the change
from the last executed command to the new first command. If neither
reference is feasible, the simulation reports a planning failure.

These checks certify feasibility only under the conditional mean and at the
checked resolution. The scenario evaluations expose possible arrival and
constraint violations under other winds. Neither a weighted average nor a
finite collection of successful scenarios guarantees feasibility for all
Gaussian disturbances.

Three execution strategies use paired physical wind realizations:

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

### Recorded outcomes and sampling sensitivity

The following table is loaded from local experiment results; building the
chapter does not run the flight optimizer.

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

The artifact also varies the number of candidate parameters and the number of
wind scenarios separately. Candidate counts of $16,32,64,128$ are compared at
$L=6$; scenario counts of $1,3,6,12,24$ are compared at $K=32$. Four independent
planning seeds are used at the initial full-trip decision. Each resulting
plan is evaluated on 128 fresh, shared wind futures that no optimizer saw.
This separates variation in the candidate search from variation in the
estimated expected cost. Two projected updates need not improve monotonically
with either sample count, and this first-decision check does not establish
convergence of the entire feedback policy.

In the recorded sweep, increasing $K$ from $16$ to $128$ at $L=6$ changes
mean held-out cost from about $57{,}396$ to $50{,}110$ kg-equivalent, with
sample standard deviations across planning seeds of $10{,}008$ and $7{,}338$.
These values include large penalties for rare unsuccessful wind futures;
they are not fuel consumption. At $K=32$, increasing $L$ from $1$ to $24$
changes the mean from $58{,}353$ to $51{,}319$, but intermediate values are
nonmonotone. The study shows sensitivity to both sampling budgets rather than
establishing convergence.

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

MPPI improves a control sequence by projecting exponentially weighted
trajectories onto a tractable distribution. Its importance weights account
for the proposal used to generate those trajectories. For matched diffusions
with quadratic control effort, the path-integral representation follows from
a logarithmic transformation of HJB. The Brownian examples check that
representation against analytic and numerical reference solutions. The
aircraft example uses a broader stochastic shooting formulation, estimating
each candidate's expected cost over wind futures before weighting candidates.

The Brownian and aircraft examples use the same computational pattern:
simulate futures, correct for the sampling distribution, and project a weighted
population back to a tractable control family. Their mathematical guarantees
differ. Covariance matching gives the Brownian example its connection to
optimal stochastic feedback; the aircraft example relies on numerical
validation and comparisons within its chosen trajectory family. These examples also motivate the
[receding-horizon control chapter](receding-horizon-control.md): executing a
short part of a plan and replanning from observations can turn any suitable
trajectory optimizer into a feedback controller. MPC studies that execution
rule, including horizon choice and feasibility, separately from the sampling
method used here.

## Computational Sources

The MPPI experiments use small numerical modules and read precomputed results
when the book is built. The Brownian sampler and its reference calculation
have separate implementations, while the aircraft evaluator averages each
candidate's costs over a shared ensemble of possible wind futures.

- {download}`Gaussian importance weights and mean updates <code/mppi_control.py>`
- {download}`Brownian dynamics, path sampler, and reference solutions <code/brownian_mppi.py>`
- {download}`Brownian experiment and figure generator <scripts/build_brownian_mppi_artifacts.py>`
- {download}`Aircraft model, wind scenarios, and trajectory optimizer <code/aircraft_mppi.py>`
- {download}`Aircraft experiment and figure generator <scripts/build_aircraft_mppi_artifacts.py>`
- {download}`ERA5 numerical snapshot <data/aircraft/era5_wind.npz>` and
  {download}`provenance and checksums <data/aircraft/era5_wind.json>`

The experiment scripts record the physical parameters, planner settings,
seeds, and validation diagnostics alongside their results. The
[Brownian replay](interactive/brownian-mppi.html) and
[aircraft replay](interactive/aircraft-mppi.html) display recorded simulations
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

A Brownian particle has $x_0=0$, target $a=1$, $N=10$ intervals of length
$h=0.1$, effort coefficient $r=1$, terminal cost
$\tfrac\kappa2(x_N-a)^2$ with $\kappa=4$, and $\lambda=0.5$.
Its sampled inputs have covariance $\Sigma_h=\lambda/(rh)$.
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

:::{exercise} Noise matching and time discretization
:label: ex-mppi-noise-scaling

Consider $dX_s=u_s\,ds+\sigma\,dW_s$ with running effort
$\tfrac12ru_s^2$, where $r>0$ and $\sigma>0$.

**(a)** Find $\lambda$ and the control-perturbation variance needed for
$x_{k+1}=x_k+h(u_k+\epsilon_k)$ to match the diffusion.

**(b)** If $h$ is halved, how must the standard deviation of $\epsilon_k$
change? What happens to the physical variance accumulated over a fixed
time horizon if the perturbation variance is instead held fixed?

**(c)** A stochastic double integrator obeys
$dX_s=V_s\,ds$ and $dV_s=u_s\,ds+\sigma\,dW_s$.
Does its unactuated position coordinate violate noise matching? Would
adding an independent Brownian increment directly to $dX_s$ preserve it?
:::

:::{solution} ex-mppi-noise-scaling
:class: dropdown

**(a)** Matching requires $\sigma^2=\lambda/r$, hence
$\lambda=r\sigma^2$. The random position increment must have variance
$\sigma^2h$. Since it equals $h\epsilon_k$, its perturbation variance is
$\operatorname{Var}(\epsilon_k)=\sigma^2/h=\lambda/(rh)$.

**(b)** The variance doubles and the standard deviation increases by
$\sqrt2$. With fixed perturbation variance $s^2$ and duration $T$, the
accumulated position variance is $(T/h)h^2s^2=Ths^2$, which vanishes as
$h\to0$. That discretization approaches deterministic dynamics instead
of the prescribed diffusion.

**(c)** In state order $(X,V)$, take $D=(0,1)^\top$ and
$B=(0,\sigma)^\top$. Then
$BB^\top=\lambda D r^{-1}D^\top$ with the value of $\lambda$ above.
Both noise and control act on velocity, so the unactuated position causes
no problem. Independent position noise would add a positive entry to the
position variance that the right side cannot represent.
:::

:::{exercise} Certainty equivalence and nonlinear costs
:label: ex-mppi-certainty-equivalence

For $dX=u\,ds+\sqrt\nu\,dW$, minimize expected terminal cost
$\tfrac\kappa2(X_{t_f}-a)^2$ plus running effort $\tfrac r2u^2$.
Assume $r,\kappa>0$, set $\lambda=r\nu$, and write $\tau=t_f-t$.

**(a)** Verify that the value function

$$
V(t,x)=\frac{\kappa r(x-a)^2}{2(r+\kappa\tau)}
+\frac{r\nu}{2}\log(1+\kappa\tau/r)
$$

satisfies the scalar HJB equation and the terminal condition.

**(b)** Derive the optimal feedback and explain which part of the value
changes with $\nu$ when $r$ is held fixed.

**(c)** Does this example show that noise never changes an optimal
decision? Suggest a modification for which that conclusion is unsupported.
:::

:::{solution} ex-mppi-certainty-equivalence
:class: dropdown

**(a)** Write $P(t)=\kappa r/(r+\kappa\tau)$ and
$b(t)=(r\nu/2)\log(1+\kappa\tau/r)$, so
$V=P(x-a)^2/2+b$. Their derivatives are
$\dot P=P^2/r$ and $\dot b=-\nu P/2$.
Substituting $V_x=P(x-a)$ and $V_{xx}=P$ into
$V_t-V_x^2/(2r)+\nu V_{xx}/2=0$ cancels both the quadratic and constant
terms. At $t=t_f$, $P=\kappa$ and $b=0$, giving the terminal cost.

**(b)** The feedback is
$u^\star=-V_x/r=\kappa(a-x)/(r+\kappa\tau)$.
Only the state-independent term $b(t)$ depends on $\nu$. It increases
expected cost but disappears on taking the state derivative.

**(c)** The cancellation uses the quadratic value function and
control-independent additive noise. A state penalty with an obstacle,
two separated target regions, or a nonlinear drift need not preserve
that structure. For example, two passages with different widths can
require different expected control effort to counter diffusion even
when their deterministic path lengths are comparable.
:::

:::{exercise} Stochastic aircraft rollouts and expected cost
:label: ex-mppi-aircraft-expected-cost

An aircraft simulator accepts a control sequence $\mathbf U$ and a gust
history $W$, and returns cost $C(\mathbf U,W)$. The objective is
$\mathbb E_W C(\mathbf U,W)$. No noise-matching property has been
established for this simulator.

**(a)** A sampling optimizer generates one gust history per candidate
control sequence and assigns weight proportional to
$e^{-C(\mathbf U,W)/\lambda}$. Explain why collecting more such weighted
samples does not generally recover the intended expected-cost objective.

**(b)** Consider a safe candidate with constant cost $1$ and a risky
candidate with cost $0$ with probability $0.9$ and $20$ with probability
$0.1$. At $\lambda=1$, compare their expected costs and their exponential
scores $-\log\mathbb E[e^{-C}]$.

**(c)** Describe a sampling design that ranks candidate controls by an
estimate of expected cost. Why does this design still not make the
classical path-integral theorem apply to the aircraft model?
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
$\widehat J(\mathbf U)=M^{-1}\sum_{j=1}^M C(\mathbf U,W^{(j)})$.
Then use $\widehat J$ as the candidate score in the sampling optimizer.
Shared gust histories across candidates can make paired comparisons
less variable. As the number of disturbance samples increases, this
score estimates the intended expected cost. The resulting method is a
sampling optimizer for a stochastic rollout objective; it does not
establish the control-affine dynamics, quadratic effort, and covariance
matching required by the HJB transformation.
:::

## Self-checks

:::{exercise} Physical noise and sampling
:label: ex-mppi-check-noise

If the optimizer stops generating candidate trajectories, does the Brownian
particle stop fluctuating? Which source of randomness has been removed?
:::

:::{solution} ex-mppi-check-noise
:class: dropdown

The physical Brownian increments remain. Stopping the sampler removes
simulated futures used to estimate an update; it does not change the
particle's process noise.
:::

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
