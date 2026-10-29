# Model Predictive Path Integral Control

A force can steer a microscopic particle toward a target, but thermal
fluctuations make its arrival uncertain. A trajectory computed under the mean
motion can pass through a narrow opening while many realized trajectories
miss it. How can a controller use these possible futures when choosing its
next action? Model predictive path integral control (MPPI) improves a control
sequence by averaging sampled perturbations, with larger weights assigned
to lower-cost futures. The same calculation connects to stochastic optimal
control under a specific relationship between noise and control effort.

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
## A Brownian Particle and One Control Update

How can sampled motion suggest a force, and how can we check the resulting
update? The starting point is the single-shooting computation from
[Numerical Trajectory Optimization](numerical-trajectory-optimization.md):
choose an input and simulate the state it produces. Brownian motion adds a
distribution of possible outcomes to that computation.

### Physical fluctuations and simulated futures

A microscopic particle suspended in a fluid moves under both an applied
force and thermal fluctuations. In normalized units, a scalar model is

$$
dX_s=u_s\,ds+\sqrt{\nu}\,dW_s,
\qquad \nu>0.
$$

Here $u_s$ is the drift produced by the applied force, and $W_s$ is standard
Brownian motion: an increment over an interval of length $h$ has mean zero
and variance $h$, independently of earlier increments. The particle remains
stochastic when the force is fixed. A simulator draws independent increments
to represent possible futures from the same observed state. Stopping these
simulations does not stop the physical fluctuations.

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
different variance from the position increment it produces. Once a control
has been selected, the physical system receives a fresh disturbance; the
simulated increments are not predictions of that particular disturbance.

For one interval, the target is $a$ and the force is held constant. With
terminal penalty coefficient $\kappa>0$ and effort coefficient $r>0$, the
control problem is

$$
\min_u J(u),\qquad
J(u)=\mathbb E\!\left[\frac\kappa2(X_1-a)^2\right]
+\frac{rh}{2}u^2.
$$

Here $X_1$ is the random position after the interval, starting from $x_0$.
The expectation includes both the mean arrival error and its variance. The
coefficient $r$ sets the cost of the force used to reduce that error.

### Calculating the weights

Take a one-unit step from $x_0=0$ toward $a=1$, starting with nominal
control $u_0=0$. Set $h=\nu=r=1$ and $\kappa=2\log2$. Thus $c_1(x)=(\log2)(x-1)^2$. Suppose three sampled
perturbations, rounded for the calculation, are $-1,0,1$. The MPPI rule
assigns a weight proportional to $\exp[-c_1(x_1)/\lambda]$, with
$\lambda=r\nu=1$ in this example. The Gaussian sampling law will account
for control effort; the following section derives why it does not appear
again inside this weight.

The three weights are obtained by evaluating the terminal cost and
normalizing their sum:

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
three trajectories. It does not simply choose the best sampled input, which
would be $1$. The Gaussian sampling frequencies and the weighting scale $\lambda$ both
affect the average.

### Checking the update analytically

The expected quadratic cost can be evaluated without sampling. The identity
$\mathbb E[(X-a)^2]=(\mathbb EX-a)^2+\operatorname{Var}(X)$ gives

$$
J(u)=\frac\kappa2\left[(x_0+hu-a)^2+\nu h\right]
+\frac{rh}{2}u^2.
$$

The variance contributes a constant with respect to $u$. Differentiating the
remaining terms gives

$$
J'(u)=\kappa h(x_0+hu-a)+rhu,
\qquad
u^*=\frac{\kappa(a-x_0)}{r+\kappa h}.
$$

For the numerical example, this yields
$u^*=2\log2/(1+2\log2)\approx0.581$. The three-sample update $0.6$ is close,
but it is only an estimate. The next section identifies its limiting
expectation and shows why that expectation agrees with the quadratic answer.

The optimal force here is independent of $\nu$: noise raises expected cost
without changing which action minimizes it. This is **certainty equivalence**.
The example checks the sampling calculation, but it cannot establish that
uncertainty changes the optimal action. The competing passages in
@sec-brownian-passages will supply that comparison.

(sec-mppi-distribution)=
## From Trajectory Weights to MPPI

Which distribution does the weighted average estimate, and how does it account
for control effort? The one-step calculation used a zero nominal force. A
useful planner must also handle many time steps and samples centered around
an existing control sequence. This requires distinguishing the distribution
that defines the objective from the distribution used to draw samples.

### Reference and sampling distributions

For a general rollout, write
$\mathbf{x}_{k+1}=\mathbf{f}_k(\mathbf{x}_k,\mathbf{v}_k)$ and collect the
realized inputs into
$\mathbf{V}=(\mathbf{v}_0,\ldots,\mathbf{v}_{N-1})$. The initial state is
fixed, so each input sequence determines a simulated state sequence. For the
Brownian particle, $v_k=u_k+\epsilon_k$ and $f_k(x,v)=x+hv$.
Define the state and terminal cost of a rollout by

$$
S(\mathbf V)=c_N(\mathbf{x}_N)
+\sum_{k=0}^{N-1}c_k(\mathbf{x}_k).
$$

Each $c_k$ is the cost for one interval; if it comes from a continuous
running cost, its discretization includes the interval length. Control
effort will enter through the distribution over inputs, just as in the
one-step example.

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
can move as the current plan improves. For the opening example the two
distributions coincide, since $\mathbf U=0$. Later, samples can concentrate
near a nonzero plan while the reference continues to define the same
objective. For the Brownian model, these input distributions
describe the drift-plus-noise terms in the time-discrete dynamics. They
also define a sampling optimizer for a deterministic simulator; that use
alone makes no assertion about physical process noise.

### Exponentially weighting the reference

The reference describes input preferences before the state costs are
considered. Multiplying it by the exponential cost weight favors trajectories
that also approach the target:

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

For the one-step particle, $P_0=\mathcal N(0,1)$ and
$S(v)=\kappa(v-1)^2/2$. Multiplication gives

$$
p^*(v)\propto
\exp\!\left[-\frac12\bigl(v^2+\kappa(v-1)^2\bigr)\right]
\propto\exp\!\left[-\frac{1+\kappa}{2}
\left(v-\frac\kappa{1+\kappa}\right)^2\right].
$$

The weighted law is Gaussian with mean $\kappa/(1+\kappa)$, exactly the
analytic action found above. The three sampled weights approximate this
mean. For nonlinear dynamics or nonquadratic costs, the weighted law usually
has no such closed form.

### The objective defined by the weights

The optimization problem solved by this density follows by expanding a
Kullback--Leibler divergence. For densities $q,p$, define
$D_{\mathrm{KL}}(Q\|P)=\mathbb E_Q[\log(q/p)]$, where $Q$ must put no mass where $P$ has zero density. This condition is
called absolute continuity. Substituting $p^\star$ gives

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

For Brownian sampling, $\Sigma=\nu/h$ and $\lambda=r\nu$, so this term
is $\tfrac12 rh\sum_k u_k^2$. This is the interval-by-interval effort
penalty from the physical model. It also explains why $S$ contains only
state costs: adding the same effort to $S$ would count that penalty twice.

The minimizer $P^*$ ranges over probability distributions on complete input
sequences. Computing that distribution is a different problem from finding
one deterministic control sequence. To implement a controller with a stored
nominal sequence, the distribution must be approximated.

### Fitting a Gaussian mean

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
second moments. The fit matches the mean of the weighted law while keeping the chosen
covariance. It therefore retains less information than $P^*$: separate
high-weight groups can be replaced by a mean between them.

There are two optimization problems here. The first minimizes expected cost
plus $\lambda D_{\mathrm{KL}}(Q\|P_0)$ and has solution $P^*$. The second
minimizes $D_{\mathrm{KL}}(P^*\|Q_{\mathbf M})$ within a Gaussian family.
Because KL divergence is not symmetric, the second problem does not generally
solve the first problem restricted to Gaussians. The quadratic example makes
the means agree; it is not a general proof of equivalence
{cite:p}`Williams2017InformationTheoretic`.

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

A numerical check separates sampling error from an incorrect target. For a
quadratic particle with $\kappa=4$, $r=1$, $\nu=0.5$, and a one-unit horizon,
the exact current action is $0.8$. The saved 200,000-sample calculation gives
$0.803$ with the correction when sampling around the nominal force $0.4$.
Omitting the correction instead approaches $0.88$, even with arbitrarily
many samples. Exercise @ex-mppi-likelihood-correction derives both limits.

### Receding-horizon execution

Suppose the particle receives the force $0.6$ but a thermal fluctuation
moves it away from the target. Continuing with a stored future sequence
would ignore that observation. Instead, apply only the first action and use
the newly observed position as the initial condition for another planning
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

(sec-path-integral-control)=
## Path-Integral Stochastic Control

When does an exponential average over stochastic trajectories compute
an optimal feedback action?

The distributional derivation specifies a mean fit. Stochastic optimal
control asks a further question: which action minimizes expected physical
cost when later actions may depend on later observations? For the Brownian
particle these later observations matter even though the initial quadratic
action agrees with the one-step calculation. The following derivation starts
from that expected-cost problem and recovers the weighted-noise formula.

### The stochastic control problem

Let $\mathbf X_s\in\mathbb R^n$ follow the control-affine diffusion

$$
d\mathbf X_s=
[\mathbf a(s,\mathbf X_s)+D(s,\mathbf X_s)\mathbf u_s]\,ds
+B(s,\mathbf X_s)\,d\mathbf W_s.
$$

The drift $\mathbf a$ describes uncontrolled motion, $D$ maps the
$m$-dimensional control into state motion, and $B$ maps Brownian noise
into state fluctuations. For the scalar particle, these objects are $\mathbf a=0$, $D=1$, and
$B=\sqrt\nu$. Controls may depend on
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

The expectation is conditional on $\mathbf X_t=\mathbf x$ and follows the
chosen feedback controller. The infimum ranges over controllers whose
current action uses only information available at the current time.
The value $V$ is the smallest expected remaining cost from that state.

Assume the coefficients are sufficiently regular for the diffusion to exist without
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

Subtracting $V$, dividing by $\delta$, and taking the limit gives the
stochastic Hamilton--Jacobi--Bellman (HJB) equation:

$$
0=V_t+c+\mathbf a^\top\nabla V
+\frac12\operatorname{tr}(BB^\top\nabla^2V)
+\min_{\mathbf u}\left\{
\frac12\mathbf u^\top R\mathbf u+\mathbf u^\top D^\top\nabla V
\right\}.
$$

For the particle, the diffusion contribution is $\nu V_{xx}/2$.
Brownian displacements are of order $\sqrt\delta$, so their squared
displacements contribute at order $\delta$, the same order as drift and
running cost. Omitting the second derivative would replace this
stochastic recursion by a deterministic one.

Only the terms inside braces depend on the current action. Since $R$ is
positive definite, differentiating this strictly convex quadratic gives
its unique minimizer:

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

The minimized HJB equation contains both a term quadratic in $\nabla V$
and a diffusion term involving $\nabla^2V$. A logarithm can connect these
terms because its second derivative includes a squared first derivative.
Define $\Psi=e^{-V/\lambda}$, or equivalently $V=-\lambda\log\Psi$.
The positive function $\Psi$ is called the desirability: states with lower
optimal cost have larger $\Psi$. Its derivatives are

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

### Returning to the quadratic particle

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

At the initial time, setting $\tau=h$ recovers the one-step action from
@sec-mppi. The value is different because a feedback controller can respond
to observations throughout the remaining interval, whereas the one-step
problem holds its force fixed. Agreement of the current actions does not
make those two execution rules identical. Exercise
@ex-mppi-certainty-equivalence verifies the value equation directly.

### Estimating the control from noise increments

The value representation still appears to require a state derivative to
obtain the control. For the particle, however, a positive first noise
increment changes the remaining expected weight by approximately
$\Psi_x\sqrt\nu\,\Delta W$. Multiplying by $\Delta W$ and averaging
keeps this term because $\mathbb E[(\Delta W)^2]=\delta$; terms constant
in the increment average to zero. Thus the correlation of the first
increment with the future weight estimates a state derivative.

To express this calculation for several control directions, take $D$ to
have full column rank and write $B=DL$, with constant
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

The random state increment is $\sqrt h\,D_kL\boldsymbol\xi_k$, so its
covariance is $hD_kLL^\top D_k^\top=hB_kB_k^\top$, as required by the
diffusion. The input perturbation itself has covariance
$\Sigma_h=\lambda R^{-1}/h$. Halving $h$ therefore increases its standard
deviation by $\sqrt2$, while reducing the standard deviation of the physical
state increment by $\sqrt2$. The two scalings refer to different quantities.
With $\lambda\Sigma_h^{-1}=hR$, approximating the integrals in
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

The same weighted perturbation now has two derivations. Their conclusions
can be compared by specifying the object each calculation returns:

| Calculation | Result before numerical approximation | Approximation used in computation |
| --- | --- | --- |
| Gibbs weighting and Gaussian fitting | Mean of $P^*$, an exact fit within the fixed-covariance Gaussian family | Finite importance samples estimate that mean; fitting discards other features of $P^*$ |
| Matched stochastic control | Optimal current feedback action from the limiting first-noise expectation | Time discretization and finite samples approximate the conditional expectation |

The distributional mean describes an entire open-loop sequence. The
stochastic-control formula gives a current feedback action conditioned on a
specified state. Its future feedback actions require the future states as
well. For general nonlinear constrained models, neither Gaussian averaging
nor finite sampling guarantees global optimality or feasibility.

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

The horizontal coordinate in the figures is time, and the vertical
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

The full passage requires a sequence of successful positions. To isolate
the decision about when to choose a route, first replace that duration by a
single crossing time $t_s<T$. The particle
pays a lump penalty $F$ if $X_{t_s}$ lies outside a union of intervals
$\mathcal A=\bigcup_j[a_j,b_j]$, and a terminal cost $\kappa X_T^2/2$.
The diffusion and effort remain

$$
dX_s=u_s\,ds+\sqrt\nu\,dW_s,
\qquad \frac r2u_s^2,
\qquad \lambda=\nu r.
$$

The desirability $\Psi=e^{-V/\lambda}$ is the expected exponential weight
under uncontrolled Brownian motion, as derived in @sec-path-integral-control. Conditioning
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
normalized Gaussian in $y$. Write $\delta=t_s-t$ for the time until
crossing and $A=r/\kappa+T-t_s$ for the remaining time after crossing plus
the effort-to-terminal-cost ratio. The Gaussian product has mean and variance

$$
\beta=\frac A{A+\delta},\qquad m=\beta x,\qquad
s^2=\nu\delta\beta.
$$

The terminal cost pulls its mean toward zero: $0<\beta<1$. Under this
cost-weighted Gaussian, the probability of lying in an opening is

$$
P_{\mathcal A}(x)=\sum_j\left[
\Phi\!\left(\frac{b_j-m}{s}\right)
-\Phi\!\left(\frac{a_j-m}{s}\right)\right],
$$

where $\Phi$ is the standard normal cumulative distribution function.
Inside an opening, the crossing weight is one; outside, it is
$e^{-F/\lambda}$. Averaging these two values gives

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
same state cost and control effort, using $x_{k+1}=x_k+hu_k$.
This baseline asks what is lost by replacing the future diffusion by its mean. It applies its
first action to the observed noisy state and recomputes the action after each
step. Both methods therefore use feedback; they differ in whether planning
accounts for the future diffusion. The deterministic position grid has spacing
$0.0025$, corresponding to action spacing $0.1$, with a numerical action limit
of $24$. The reference convolution uses spacing $0.005$ on $[-6,6]$.

### Passage choice, cost, and failures

Does accounting for diffusion change which passage is used, and does that
choice reduce realized cost? The comparison uses 128 independent physical
disturbance sequences, paired across controllers. Pairing gives the methods
the same fluctuations, so an apparent improvement cannot be explained by one
method receiving an easier collection of disturbances.

| Physical noise | Sampled path-integral cost | Mean-dynamics cost | Numerical-reference cost |
| --- | ---: | ---: | ---: |
| $\nu=0.015$ | $0.221\pm0.085$ | $5.310\pm0.914$ | $0.197\pm0.078$ |
| $\nu=0.3$ | $1.808\pm0.289$ | $11.303\pm1.184$ | $1.767\pm0.279$ |

The entries are means with normal-approximation 95% confidence intervals
across trials. The sampled controller and the independent numerical
reference give similar mean costs. Mean-dynamics planning has larger cost
at both noise levels, even though it also replans from the observed state.

At low noise, neither planning method occupies the wide passage at time
$1.5$. At high noise, their choices separate:

| Controller, $\nu=0.3$ | Wide-passage trials | Wilson 95% interval for the proportion |
| --- | ---: | ---: |
| Sampled path-integral feedback | $126/128$ | $[0.945,0.996]$ |
| Mean-dynamics planning | $32/128$ | $[0.183,0.332]$ |

Occupancy includes every trial, including those that incur a penalty.
Accounting for the stronger diffusion favors the wider route despite its
detour. This is the effect that the quadratic target alone could not show.

A finite penalty still allows failed passages. A trial fails if any checked
position lies outside both openings during $1<t\leq2$:

| Physical noise | Sampled path-integral failures | Mean-dynamics failures |
| --- | ---: | ---: |
| $\nu=0.015$ | $17/128$ | $94/128$ |
| $\nu=0.3$ | $28/128$ | $122/128$ |

At high noise, mean control effort is $1.142$ for sampled feedback and
$1.531$ for mean-dynamics planning. The cost difference includes both the
effort and the accumulated penalty for positions outside the passages.

:::{figure} _static/brownian_mppi/brownian-outcomes.svg
:label: fig-brownian-outcomes
:width: 100%

Costs and passage outcomes under paired physical disturbances. Cost bars are
95% intervals for means; proportion bars are Wilson 95% intervals. The numerical
reference and the sampled controller have similar outcomes. The finite
penalty does not impose a probability-of-failure guarantee.
:::

### Sampling error and discretization error

The reference solution lets us vary numerical accuracy separately from
physical noise. At the high-noise initial state, increasing the candidate count from 256 to 16384 reduces the
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

Consider an Airbus A320 flying from Montréal–Trudeau (CYUL) to Toronto–Pearson
(CYYZ). The task is to choose climb, cruise, descent, and flight duration to
use little fuel while arriving within 1 km horizontally and 30 m vertically
of the destination. A favorable wind at one altitude may justify climbing,
but its later evolution is uncertain. How can the sampled-control update
account for those winds when the Brownian noise-matching condition does not
hold?

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

Every candidate receives the same ensemble of wind futures. This allows
paired cost differences within a planning update, just as paired physical
disturbances allowed controller comparisons in the passage experiment.
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

The aircraft model has no established covariance–effort matching relation.
Its update therefore uses the distributional construction from
@sec-mppi-distribution, with $\widehat C$ as its score. The optimal-feedback
conclusion of @sec-path-integral-control does not follow. This is the
distinction between the two derivations discussed by
{cite:t}`Williams2017InformationTheoretic`.

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

For diffusions with quadratic effort and matched control and noise
covariances, the logarithmic HJB transformation gives a second derivation.
It expresses optimal current feedback as a weighted expectation of initial
noise increments. The quadratic particle checks the calculation, while the
passage geometry shows how physical uncertainty can change the chosen route.
The aircraft application separates control exploration from wind uncertainty
and averages wind costs before assigning a candidate its weight.

The controller still needs an execution rule. Applying one action, observing
the result, and solving again provides feedback even though each stored
sequence is open loop. [Receding-horizon control](receding-horizon-control.md)
develops that rule further, including the choice of horizon, terminal
conditions, and feasibility safeguards. These questions apply whether the
planner uses derivatives or sampled trajectories.

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
$\widehat C(\mathbf U)=L^{-1}\sum_{\ell=1}^L C(\mathbf U,W^{(\ell)})$.
Then use $\widehat C$ as the candidate score in the sampling optimizer.
Shared gust histories across candidates can make paired comparisons
less variable. As the number of disturbance samples increases, this
score estimates the intended expected cost. The resulting method is a
sampling optimizer for a stochastic rollout objective; it does not
establish the control-affine dynamics, quadratic effort, and covariance
matching required by the HJB transformation.
:::

:::{exercise} Comparing passage controllers with paired trials
:label: ex-mppi-paired-passages

Use the saved {download}`passage trials <artifacts/brownian_mppi/trials.npz>`.
The arrays named `<noise>_<method>_cost` contain one cost per trial, with
`noise` equal to `low_noise` or `high_noise` and `method` equal to
`path_integral` or `mean_dynamics`. Matching indices use the same physical
Brownian increments.

**(a)** For each noise level, compute the trial differences
$d_j=C_j^{\mathrm{PI}}-C_j^{\mathrm{mean}}$. Report their sample mean and the
normal-approximation 95% interval
$\bar d\pm1.96s_d/\sqrt n$, where $s_d$ is the sample standard deviation.

**(b)** Explain why this interval answers a different question from an
interval for the cost of one future flight or particle trajectory. Why
should the difference be formed before its standard error is computed?

**(c)** Design a follow-up experiment that distinguishes a sampling error
from an error in time discretization. Specify what is held fixed in each
comparison and which reference solution is used. State what the saved
failures prevent you from claiming about the controller.
:::

:::{solution} ex-mppi-paired-passages
:class: dropdown

**(a)** The calculation uses the within-trial difference:

```python
import numpy as np

with np.load("artifacts/brownian_mppi/trials.npz") as trials:
    for noise in ("low_noise", "high_noise"):
        d = (trials[f"{noise}_path_integral_cost"]
             - trials[f"{noise}_mean_dynamics_cost"])
        half_width = 1.96 * d.std(ddof=1) / np.sqrt(d.size)
        print(noise, d.mean(), half_width)
```

The mean differences are approximately $-5.090\pm0.883$ at low noise and
$-9.496\pm1.070$ at high noise. Negative values favor path-integral feedback.

**(b)** This interval describes uncertainty in the estimated mean cost
difference across independent trials. An individual realized cost can be far
from that mean. Pairing also contributes a covariance term:
$\operatorname{Var}(C^{\mathrm{PI}}-C^{\mathrm{mean}})
=\operatorname{Var}(C^{\mathrm{PI}})+\operatorname{Var}(C^{\mathrm{mean}})
-2\operatorname{Cov}(C^{\mathrm{PI}},C^{\mathrm{mean}})$.
Computing the differences preserves the covariance produced by shared
physical disturbances; treating the two arrays as independent discards it.

**(c)** Hold the model, state, time step, and reference grid fixed while
varying candidate count across independent sampling seeds. Compare the first
action with the backward-convolution action at that time step. Separately,
vary the time step in the reference calculation while keeping physical
coefficients fixed, including $\lambda=r\nu$ and the corresponding
perturbation scaling. These comparisons isolate Monte Carlo error and time
resolution. The observed failures and unchecked crossings between time
steps rule out a claim of guaranteed passage feasibility.
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
