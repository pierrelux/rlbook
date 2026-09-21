(sec-path-integral-control)=
# Path-Integral Stochastic Control

Stochastic dynamic programming chooses a feedback action by minimizing
immediate cost plus expected continuation cost. In continuous time, that
recursion becomes a partial differential equation for the value function.
Can forward simulations compute the optimal action without solving this
equation over a grid of states?

For a class of systems with quadratic control effort and matching physical
noise, a logarithmic transformation turns the value equation into a linear
one. Its solution is an expectation over trajectories, which can be
estimated by sampling. This gives conditions under which the weighted
updates introduced in the [MPPI chapter](model-predictive-path-integral-control.md)
compute an optimal current feedback action. A particle moving through two
competing passages then shows how physical uncertainty can change the
preferred route.

## Learning Goals

After working through the derivation and experiments, you should be able to:

- Derive the stochastic Hamilton--Jacobi--Bellman equation from the
  short-time Bellman recursion.
- State the noise and control-cost matching condition and use it to obtain
  a path-integral representation of the value function.
- Estimate a current feedback action from weighted noise increments,
  including the correction for a controlled sampling process.
- Distinguish an optimal feedback action from a Gaussian fit to a
  distribution over open-loop control sequences.
- Compare stochastic and mean-dynamics controllers using paired
  disturbances, numerical reference solutions, and discretization checks.

## Prerequisites

This chapter uses the Bellman recursion and state-dependent policies from
[Finite-Horizon Dynamic Programming](finite-horizon-dp.md) and
[Stochastic Dynamic Programming](stochastic-dp.md). Gaussian increments are
reviewed from [Stochastic Dynamics and Partial Observation](stochastic-dynamics-observation.md).
The comparison with MPPI uses the Gaussian mean fit and importance-sampling
correction in @sec-mppi-distribution. The continuous-time value equation
and its transformation are derived below.

## Stochastic Feedback Control

How does the Bellman recursion change when controls and disturbances act
continuously in time? A scalar model with Gaussian increments gives the
noise scaling needed to take the short-time limit. The resulting value
function accounts for future actions responding to future observations.

### A model with physical noise

Consider a scalar position whose control $u_k$ sets its average velocity
over an interval of length $h$. Suppose the motion also has an independent
Gaussian displacement during each interval:

$$
x_{k+1}=x_k+hu_k+\sqrt{\nu h}\,\xi_k,
\qquad \xi_k\sim\mathcal N(0,1),\qquad \nu>0.
$$

Here $\nu$ is the variance of the physical displacement per unit time.
Over $N$ intervals, the accumulated noise variance is $N\nu h$.
Thus subdividing a fixed duration into smaller steps preserves the total
variance. The factor $\sqrt h$ gives this property: using $h$ instead
would make the total noise variance vanish as the steps become smaller.

Brownian motion is the continuous-time process that supplies these
independent Gaussian increments. If $W_s$ denotes standard Brownian motion,
then $W_{s+h}-W_s\sim\mathcal N(0,h)$. The continuous-time model is written

$$
dX_s=u_s\,ds+\sqrt\nu\,dW_s.
$$

For a control held constant over one interval, integrating this equation
gives the Gaussian transition above exactly. The notation $dW_s$ denotes
an increment of the random process. It is not an ordinary velocity that
can be evaluated and multiplied by a time step.

This physical randomness remains even when the control is fixed. A
simulator draws independent increments to represent possible futures from
the observed state; those draws do not predict the particular disturbance
the system will experience. To use the rollout notation from
@sec-mppi-distribution, write
the same transition as $x_{k+1}=x_k+h(u_k+\epsilon_k)$, where
$\epsilon_k=\sqrt{\nu/h}\,\xi_k$. Its variance is $\nu/h$ because it is
multiplied by $h$ to produce the physical displacement. In this model,
physical noise determines the perturbation covariance used in the
path-integral derivation.

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

The stochastic Bellman recursion compares immediate cost with expected
optimal continuation cost. Over a short interval $\delta$, its
continuous-time form is:

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

## Path-Integral Representation

Can the nonlinear value equation be transformed into an expectation that
we can estimate with forward simulations? The noise-matching condition
allows a logarithmic change of variables to cancel the term quadratic in
the value gradient. The resulting linear equation has a probabilistic
solution.

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

The linear equation has a probabilistic solution. Let $P_0$ be the law of
trajectories generated by the uncontrolled process
$d\mathbf X_s=\mathbf a\,ds+B\,d\mathbf W_s$. Along one such trajectory,
consider

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

### Quadratic cost and optimal feedback

For the scalar particle with terminal cost
$c_f(x)=\tfrac\kappa2(x-a)^2$, with $\kappa>0$, and no running state cost,
the uncontrolled endpoint is $X_{t_f}\sim\mathcal N(x,\nu\tau)$, where
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
For this quadratic example it does not change the feedback law. This is
**certainty equivalence**. Nonquadratic state costs or barriers are needed
to observe decisions that depend on the diffusion strength.

At $x=0$, $a=1$, $r=1$, and $\tau=1$, the current action is
$\kappa/(1+\kappa)$, the mean found in the deterministic quadratic check
in @sec-mppi-distribution. Here the controller also responds to later
observations by reevaluating $u^\star(t,x)$ at the new state and time.
Agreement of the initial actions does not make the execution rules
identical. Exercise @ex-mppi-certainty-equivalence verifies the value
equation directly.

## Sampling the Optimal Action

How can sampled trajectories estimate the feedback action without
numerically differentiating the value function? The first noise increment
changes both the next state and the expected future cost weight. Their
correlation supplies the derivative needed for the current action.

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
diffusion, and denote the resulting trajectory law by
$Q_{\bar{\mathbf u}}$. Girsanov's change-of-measure formula gives

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
expansion in @sec-mppi-distribution: the accumulated squared mean shift
supplies the first integral and the mean-noise cross terms supply the second.

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

The weighted perturbation has the same form as the update in
@sec-mppi-distribution. That derivation fits a Gaussian to the Gibbs law
$P^*$ with density proportional to $e^{-S/\lambda}p_0$, where $p_0$ is a
reference density over input sequences. Its result and the stochastic
control result describe different objects:

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

The mean-dynamics controller often waits near an opening's edge. Which
term in its predicted objective assigns a cost to remaining close to that
edge, and which future events does that prediction omit? The stochastic
controller accounts for the cost of correcting future fluctuations, which
can favor a wider passage even when reaching it requires more travel.

## Summary and Outlook

The stochastic HJB equation is the continuous-time Bellman equation for
feedback control under physical noise. With quadratic effort and matching
control and noise directions, the logarithmic transformation turns this
nonlinear equation into a linear one. The transformed value is an
exponential expectation over uncontrolled trajectories, and correlations
with the first noise increment give the optimal current action.

Importance sampling allows those trajectories to be drawn around a useful
control. Time discretization then produces the weighted-perturbation form
used by MPPI. The continuous-time result concerns an action conditioned on
the current state; finite samples and a finite time step introduce errors,
and a stored sequence still requires replanning to supply feedback.

The quadratic example has certainty equivalence, whereas competing
passages make future fluctuations affect the route choice. Paired trials
and numerical reference calculations separate that modeling effect from
sampling and discretization errors. The matching assumption also limits
the theory's scope: the aircraft optimizer in @sec-aircraft-mppi uses a
general distributional fit to estimated expected costs, which by itself
does not establish optimal feedback.

## Computational Sources

The passage experiments read precomputed results when the book is built.
The path sampler and reference calculation have separate implementations:

- {download}`Brownian dynamics, path sampler, and reference solutions <code/brownian_mppi.py>`
- {download}`Brownian experiment and figure generator <scripts/build_brownian_mppi_artifacts.py>`
- {download}`Experiment parameters and results <artifacts/brownian_mppi/results.json>`
- {download}`Paired trial arrays <artifacts/brownian_mppi/trials.npz>`

The [Brownian replay](interactive/brownian-mppi.html) displays the recorded
simulations without a Python kernel. Regeneration commands are in the
repository README.

## Exercises

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
interval for the cost of one future particle trajectory. Why
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
