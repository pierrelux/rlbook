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

# iLQR and Differential Dynamic Programming

A boat approaching a quay must arrive in the right place, face along the
berth, and lose its remaining momentum. A sequence of thrust commands determines
all three outcomes. The [shooting formulation](numerical-trajectory-optimization.md)
turns this task into an optimization over those commands: simulate a candidate
sequence, measure its cost, and improve it. How can an optimizer use the
time structure of the simulation when computing that improvement?

Linearizing the dynamics along the current trajectory and quadratically
approximating its costs gives a problem whose controls can be eliminated
backward, one time step at a time. The resulting correction is then tested
through the nonlinear simulator. Repeating these operations gives the
**iterative linear-quadratic regulator**, or **iLQR**. Keeping additional
curvature from the dynamics gives **differential dynamic programming**, or
**DDP**. Both methods compute corrections around a nominal trajectory. Their
backward recursions follow from substitution and quadratic minimization.

## Learning Goals

After working through the derivations and docking example, you should be able to:

- Construct a local quadratic optimal-control problem around a feasible rollout.
- Eliminate its controls backward and recover an affine correction at each step.
- Implement nonlinear trial rollouts with line search and regularization.
- Identify the dynamics-curvature terms that distinguish DDP from iLQR.
- Assess a docking plan using its cost, arrival velocity, and hull clearance.

## Prerequisites

The chapter uses the [finite-horizon optimal-control formulation](discrete-time-optimal-control.md),
[single shooting](numerical-trajectory-optimization.md), and first and second
derivatives. The [SQP discussion](#sec-sqp-newton) explains successive quadratic
subproblems, and the [initial-value-problem appendix](appendix_ivps.md) reviews
numerical integration. Neither dynamic programming nor a previous derivation
of LQR is required.

(sec-boat-docking)=
## A Docking Problem

What distinguishes arriving at a berth from merely passing through it?
Position alone does not specify a successful docking maneuver. The boat's
heading must agree with the quay, and its translational and angular velocities
must be small enough that it remains near the berth after arrival.

Consider a two-metre boat moving in the horizontal plane. Its state and control
are

$$
\mathbf x=(p_x,p_y,\psi,v_x,v_y,\omega),\qquad
\mathbf u=(u_L,u_R)\in[-1,1]^2.
$$

The position and velocity are measured in fixed world coordinates, $\psi$ is
the heading, and $\omega$ is the angular velocity. Each control is a fraction
of maximum thrust. Positive commands push forward and negative commands push
backward. Unequal thrusts turn the boat. There is no directly commanded
sideways force, so an approach with sideways momentum requires a maneuver.

Let $\mathbf e(\psi)=(\cos\psi,\sin\psi)^\top$ point along the boat and
$\mathbf n(\psi)=(-\sin\psi,\cos\psi)^\top$ point across it. The simplified
inertial model is

$$
\begin{aligned}
\dot{\mathbf p}&=\mathbf v,&\qquad \dot\psi&=\omega,\\
m\dot{\mathbf v}
&=F_{\max}(u_L+u_R)\mathbf e
  -d_\parallel(\mathbf e^\top\mathbf v)\mathbf e
  -d_\perp(\mathbf n^\top\mathbf v)\mathbf n,\\
I\dot\omega&=bF_{\max}(u_R-u_L)-d_\omega\omega.
\end{aligned}
$$

The drag opposes velocity along each boat axis, with stronger resistance to
sideways motion. Thrust changes acceleration, so releasing the controls does
not stop the boat immediately. These equations are a synthetic teaching model
for calm water, with neither contact forces nor waves.

| Quantity | Value |
| :--- | ---: |
| Mass $m$ | $50\ \mathrm{kg}$ |
| Angular inertia $I$ | $20\ \mathrm{kg\,m^2}$ |
| Thruster lever arm $b$ | $0.4\ \mathrm m$ |
| Maximum force per thruster $F_{\max}$ | $25\ \mathrm N$ |
| Longitudinal drag $d_\parallel$ | $8\ \mathrm{N\,s/m}$ |
| Lateral drag $d_\perp$ | $30\ \mathrm{N\,s/m}$ |
| Angular drag $d_\omega$ | $10\ \mathrm{N\,m\,s}$ |
| Rectangular hull length and width | $2\ \mathrm m$, $0.8\ \mathrm m$ |

The quay occupies $p_y\leq0$, and the desired arrival state is
$\mathbf x_\star=(0,1,0,0,0,0)$. The boat should lie parallel to the quay with
its center one metre from the edge. The angled approach starts from
$(-8,5,-0.5,0.2,-0.1,0)$. The sideways-drift case changes only $v_y$ to
$-0.5\ \mathrm{m/s}$. Angles are in radians. Both experiments use $N=200$
piecewise-constant controls, each held for $h=0.1\ \mathrm s$, for a total
of 20 seconds. One fourth-order Runge--Kutta step gives the discrete transition
$\mathbf x_{t+1}=\mathbf f_t(\mathbf x_t,\mathbf u_t)$.

To discourage the hull from approaching the quay, let $y_j(\mathbf x)$ be the
world-coordinate height of rectangular corner $j$. With a desired buffer
$a=0.2\ \mathrm m$ and smoothing length $s=0.08\ \mathrm m$, define

$$
\rho(z)=s\log(1+\exp(z/s)),\qquad
\mathcal P(\mathbf x)=200\sum_{j=1}^4\rho(a-y_j(\mathbf x))^2.
$$

This is a smooth penalty for violating the buffer. Its finite weight permits
violations, so the final trajectories will also undergo a separate geometric
clearance check. The full rectangle is used for that check, including the area
outside the pointed boat icon in the figures.

For $\mathbf e_p=\mathbf p-(0,1)^\top$, the running and terminal costs are

$$
\begin{aligned}
c_t(\mathbf x,\mathbf u)
&=h\big[0.03\|\mathbf e_p\|^2+0.1(1-\cos\psi)
       +0.1\|\mathbf v\|^2+0.1\omega^2
       +0.2\|\mathbf u\|^2+\mathcal P(\mathbf x)\big],\\
\phi(\mathbf x)
&=300\|\mathbf e_p\|^2+100(1-\cos\psi)
       +200\|\mathbf v\|^2+100\omega^2+\mathcal P(\mathbf x).
\end{aligned}
$$

These numerical weights apply to the stated SI coordinates and dimensionless
thrust commands. The periodic heading penalty assigns the same cost to angles
that differ by a full turn. The simulator retains an unwrapped angle; it does
not insert a discontinuity at $\pm\pi$ into the differentiated transition.

The initial control sequence is zero, so the boat coasts and slows under drag.
It stops far from the berth. Successive control updates can first bring it
closer and then adjust the arrival heading and velocity.

:::{figure} _static/boat_docking/iterations.svg
:label: fig-boat-iterations
:width: 100%

Accepted iLQR iterates for the angled approach. Every curve is a nonlinear
rollout from the same initial state. Boat outlines are spaced four seconds
apart; their overlap indicates slow motion. The dotted outline marks the
desired berth. The first update reduces the position error substantially,
while later updates refine the turn and stopping maneuver.
:::

(sec-local-quadratic-trajectory)=
## Local Linear Dynamics and Quadratic Costs

How can one improve a complete thrust sequence without optimizing the nonlinear
simulation from scratch at every trial? Begin with the finite-horizon problem

$$
\min_{\mathbf u_0,\ldots,\mathbf u_{N-1}}
J(\mathbf U)=\phi(\mathbf x_N)+\sum_{t=0}^{N-1}c_t(\mathbf x_t,\mathbf u_t),
\qquad
\mathbf x_{t+1}=\mathbf f_t(\mathbf x_t,\mathbf u_t),\quad
\mathbf x_0=\mathbf x_{\mathrm{init}}.
$$

First set aside the control bounds to derive the unconstrained correction.
Roll out a nominal sequence $\bar{\mathbf U}$ to obtain
$\bar{\mathbf x}_{t+1}=\mathbf f_t(\bar{\mathbf x}_t,\bar{\mathbf u}_t)$.
Write changes from this trajectory as $\delta\mathbf x_t$ and
$\delta\mathbf u_t$. The first-order dynamics are

$$
\delta\mathbf x_{t+1}=A_t\delta\mathbf x_t+B_t\delta\mathbf u_t,
\qquad
A_t=\mathbf f_{x,t},\quad B_t=\mathbf f_{u,t},\quad
\delta\mathbf x_0=\mathbf0.
$$

All derivatives are evaluated on the nominal trajectory. No constant defect
appears because that trajectory satisfies the discrete dynamics. A collection
of independently chosen state guesses would generally require such a defect.

Let $\delta\mathbf z_t=(\delta\mathbf x_t,\delta\mathbf u_t)$ stack the
local changes. The second-order expansion of the running cost is

$$
c_t(\bar{\mathbf x}_t+\delta\mathbf x_t,
    \bar{\mathbf u}_t+\delta\mathbf u_t)
\approx \bar c_t+
\begin{bmatrix}c_{x,t}\\c_{u,t}\end{bmatrix}^{\!\top}\delta\mathbf z_t
+\frac12\delta\mathbf z_t^\top
\begin{bmatrix}c_{xx,t}&c_{xu,t}\\c_{ux,t}&c_{uu,t}\end{bmatrix}
\delta\mathbf z_t.
$$

The linear terms measure the remaining incentive to move away from the nominal
trajectory. They do not vanish merely because the costs are quadratic. The
cross term describes how the marginal cost of a control changes with the
state. Even if the original running cost has no cross term, eliminating later
states will generally create one.

Expand the terminal cost in the same way, with linear coefficient
$p_N=\phi_x(\bar{\mathbf x}_N)$ and quadratic coefficient
$P_N=\phi_{xx}(\bar{\mathbf x}_N)$. Minimizing the sum of these quadratic
expressions subject to the linearized transitions gives a local quadratic
program. Its state variables couple only neighboring time steps. The
[SQP construction](#sec-sqp-newton) gives a useful comparison: here we begin
with cost curvature and linearized dynamics, omitting the second derivatives
of the dynamics that enter an exact Lagrangian Hessian.

## Backward Elimination

Why eliminate the last control first? Once its starting state is fixed, that
control affects only its own stage and the terminal cost. Solving for it leaves
a smaller problem of the same form.

### A two-step calculation

Consider a scalar example with $x_0=0$, dynamics $x_{t+1}=x_t+u_t$, and cost

$$
J=\frac12u_0^2+\frac12u_1^2+\frac12(x_2-1)^2.
$$

Treat $x_1$ as given while eliminating $u_1$. Substitution of $x_2=x_1+u_1$
gives

$$
\frac12u_1^2+\frac12(x_1+u_1-1)^2
=u_1^2+(x_1-1)u_1+\frac12(x_1-1)^2.
$$

Differentiating with respect to $u_1$ gives
$2u_1+x_1-1=0$, so $u_1=(1-x_1)/2$. Substituting this expression back into
the last two terms of the cost leaves $(x_1-1)^2/4$. The remaining problem is
therefore

$$
\min_{u_0}\ \frac12u_0^2+\frac14(u_0-1)^2.
$$

Its derivative is $u_0+(u_0-1)/2$, giving $u_0=1/3$. Forward substitution then
gives $x_1=1/3$, $u_1=1/3$, $x_2=2/3$, and $J=1/6$. The final position is short
of one because the objective trades terminal error against effort; arrival
was penalized, not imposed as an equality.

The expression $u_1=(1-x_1)/2$ also carries information that a single number
would discard: if the state reaching the last step changes, the minimizing
last control changes with it. This dependence becomes the matrix feedback
correction in the general calculation.

### Eliminating a vector control

Suppose all controls after time $t$ have already been eliminated from the
local problem. Write the remaining quadratic tail as

$$
S_{t+1}(\delta\mathbf x)
=s_{t+1}+p_{t+1}^\top\delta\mathbf x
+\frac12\delta\mathbf x^\top P_{t+1}\delta\mathbf x.
$$

This is a quadratic expression in the current trajectory's perturbations.
It is obtained by elimination from a finite optimization problem, without
constructing a function over all states of the nonlinear system.

Substitute $\delta\mathbf x_{t+1}=A_t\delta\mathbf x_t+B_t\delta\mathbf u_t$
into this tail and add the stage cost. With $F_t=[A_t\ B_t]$, the new linear
and quadratic coefficients are

$$
q_t=c_{z,t}+F_t^\top p_{t+1},\qquad
H_t=c_{zz,t}+F_t^\top P_{t+1}F_t.
$$

Partition these coefficients according to the state and control coordinates.
Apart from a constant, the expression to minimize is

$$
q_x^\top\delta\mathbf x+q_u^\top\delta\mathbf u
+\frac12\delta\mathbf x^\top H_{xx}\delta\mathbf x
+\delta\mathbf u^\top H_{ux}\delta\mathbf x
+\frac12\delta\mathbf u^\top H_{uu}\delta\mathbf u.
$$

The time index is suppressed within this one-step calculation. When $H_{uu}$
is positive definite, setting the control derivative to zero gives

$$
H_{uu}\delta\mathbf u+q_u+H_{ux}\delta\mathbf x=0,
\qquad
\delta\mathbf u=k+K\delta\mathbf x,
$$

where

$$
k=-H_{uu}^{-1}q_u,\qquad K=-H_{uu}^{-1}H_{ux}.
$$

The vector $k$ changes the nominal command even at zero state deviation. The
matrix $K$ adjusts that change for a different state arriving from the earlier
steps. In code, these formulas are linear solves with $H_{uu}$; an explicit
matrix inverse is unnecessary.

Substituting the minimizing control back into the quadratic expression gives
the coefficients needed by the preceding step:

$$
\begin{aligned}
p_t&=q_x-H_{ux}^\top H_{uu}^{-1}q_u,\\
P_t&=H_{xx}-H_{ux}^\top H_{uu}^{-1}H_{ux}.
\end{aligned}
$$

The subtracted term in $P_t$ is the Schur complement associated with
eliminating the control block. Repeat this operation from $N-1$ to zero,
starting with the terminal coefficients. A subsequent forward substitution
recovers the state and control changes. For fixed state and control dimensions,
the number of these elimination steps grows linearly with the horizon.

For linear dynamics and quadratic costs, this procedure solves the original
unconstrained problem exactly when its control minimizations are well posed.
That problem is called the **linear-quadratic regulator** problem, and the
matrix recursion for $P_t$ is a **Riccati recursion**. Along a nonlinear
trajectory, $A_t$, $B_t$, and the cost derivatives describe only a local
approximation. They must be recomputed after the trajectory changes.

(sec-ilqr-rollout)=
## Nonlinear Rollouts and iLQR

What happens when the quadratic problem proposes a large change? Its predicted
states satisfy the linearized dynamics, which can differ substantially from
the nonlinear dynamics over a large displacement. Evaluate the proposed
controls through the original discrete simulator before accepting them.

Starting from the same initial state, form a nonlinear trial trajectory using

$$
\begin{aligned}
\mathbf x_0^+&=\mathbf x_{\mathrm{init}},\\
\mathbf u_t^+&=\bar{\mathbf u}_t+\alpha k_t
                  +K_t(\mathbf x_t^+-\bar{\mathbf x}_t),\\
\mathbf x_{t+1}^+&=\mathbf f_t(\mathbf x_t^+,\mathbf u_t^+).
\end{aligned}
$$

The step size $\alpha$ scales the feedforward change. The feedback term
responds to the state deviation that actually occurs during this trial. Its
gain is not multiplied by $\alpha$. At $\alpha=0$, induction through these
equations recovers the nominal trajectory, because each state deviation is
then zero. For an exact linear-quadratic problem, $\alpha=1$ gives the complete
solution derived by elimination.

A line search first tries $\alpha=1$, then successively smaller values. Each
trial has its own nonlinear states and cost. The implementation uses
$1,1/2,\ldots,2^{-11}$ and accepts a strictly lower finite cost with a small
sufficient-decrease check. If

$$
d_1=\sum_t q_{u,t}^\top k_t,\qquad
d_2=\frac12\sum_t k_t^\top H_{uu,t}k_t,
$$

the backward pass estimates a reduction
$\widehat\Delta J(\alpha)=-\alpha d_1-\alpha^2d_2$.
Acceptance requires
$J(\bar{\mathbf U})-J(\mathbf U^+)\geq10^{-4}\max(\widehat\Delta J(\alpha),0)$
as well as a strict decrease. This estimate guides acceptance; it does not
replace evaluation of the original objective.

The control curvature may be indefinite or nearly singular. Replace the
matrix used in the control solve by $H_{uu}+\mu I$, where $\mu>0$. Increasing
$\mu$ makes the correction more conservative and can make the solve positive
definite. The code checks a Cholesky factorization, increases $\mu$ after a
failed backward pass or line search, and decreases it after an accepted step.
The numerical defaults start at $10^{-4}$, multiply by ten after failure,
and divide by three after acceptance, with a floor of $10^{-9}$.

When regularization changes $k$ and $K$, the simplified Schur-complement
formulas above no longer describe substitution into the original quadratic
model. The implementation therefore uses the full expressions

$$
\begin{aligned}
p_t&=q_x+H_{ux}^\top k+K^\top q_u+K^\top H_{uu}k,\\
P_t&=H_{xx}+H_{ux}^\top K+K^\top H_{ux}+K^\top H_{uu}K.
\end{aligned}
$$

Here $H_{uu}$ is the original curvature block; regularization was used to
choose the correction. These expressions also remain applicable when some
controls are fixed at their bounds.

Alternating local linear-quadratic approximation, backward elimination, and
nonlinear rollout gives iLQR {cite:p}`li2004iterative`. The iteration uses
second derivatives of the costs but only first derivatives of the dynamics.
It is often described as a Gauss--Newton approximation to DDP. That terminology
should not obscure what is approximated: the omitted terms come from dynamics
curvature. If a nonlinear cost is itself a sum of squared residuals, replacing
its Hessian by a residual-Jacobian product would be a further approximation;
the docking code uses the exact cost Hessian.

````{prf:algorithm} Successive linear-quadratic trajectory optimization
:label: alg-trajectory-ilqr

**Input:** Discrete transition, running and terminal costs, initial state, and a feasible initial control sequence.

1. Roll out the current controls and evaluate their nonlinear cost.
2. Differentiate the transition and costs along this rollout.
3. Starting from the terminal quadratic expansion, eliminate controls backward to obtain $k_t$ and $K_t$. Increase regularization and retry if a control solve fails.
4. Test decreasing step sizes using nonlinear forward rollouts with the affine corrections.
5. Accept a trial only when its actual cost decreases sufficiently. Otherwise increase regularization and retry the backward pass.
6. Repeat until the control-gradient residual is small or a declared iteration or regularization limit is reached.

**Output:** The last accepted trajectory, its controls and local gains, an iteration history, and an explicit termination status.
````

The returned gains describe a neighborhood of the final nominal trajectory.
The docking animation below executes one stored plan. The gains used inside
the optimizer do not imply that the animation replans as the boat moves.

(sec-ddp-curvature)=
## Dynamics Curvature and DDP

Which second-order effects are lost when the transition is linearized?
A turning command changes the direction in which a later thrust acts.
Products of state and control changes can therefore contribute to the next
state, even if they are absent from a first-order approximation.

For component $i$ of the transition, retain its second-order expansion:

$$
\delta x_{t+1,i}
\approx (F_t\delta\mathbf z_t)_i
+\frac12\delta\mathbf z_t^\top f^i_{zz,t}\delta\mathbf z_t.
$$

Substitute this expression into the same quadratic tail $S_{t+1}$. Its
linear term $p_{t+1}^\top\delta\mathbf x_{t+1}$ now contributes an additional
quadratic term. Its quadratic term contributes
$\tfrac12\delta\mathbf z_t^\top F_t^\top P_{t+1}F_t\delta\mathbf z_t$;
products involving the second-order part of the transition there are of order
three or higher and are discarded. Consequently,

$$
\begin{aligned}
q_t^{\mathrm{DDP}}&=c_{z,t}+F_t^\top p_{t+1},\\
H_t^{\mathrm{DDP}}&=c_{zz,t}+F_t^\top P_{t+1}F_t
                  +\sum_{i=1}^{n_x}p_{t+1,i}f^i_{zz,t}.
\end{aligned}
$$

The extra contraction weights curvature of each next-state component by its
linear coefficient in the eliminated tail. At a given backward stage, this is
the second-order chain rule for composing the transition with that tail.
In block form, the differences from iLQR are

$$
\begin{aligned}
H_{xx}^{\mathrm{DDP}}&=c_{xx}+A^\top P_{t+1}A
                    +\sum_i p_{t+1,i} f^i_{xx},\\
H_{ux}^{\mathrm{DDP}}&=c_{ux}+B^\top P_{t+1}A
                    +\sum_i p_{t+1,i} f^i_{ux},\\
H_{uu}^{\mathrm{DDP}}&=c_{uu}+B^\top P_{t+1}B
                    +\sum_i p_{t+1,i} f^i_{uu}.
\end{aligned}
$$

Setting those three sums to zero recovers the iLQR construction. Retaining them
and using the same quadratic elimination and nonlinear line search gives DDP
{cite:p}`tassa2014control`. The two methods share an algorithmic structure; the
curvature choice can change both the direction of a step and the amount of
regularization needed to accept it. This derivation does not require solving
a subproblem with quadratic dynamics constraints.

For the boat, even the continuous acceleration contains a state--control
cross derivative:
$\partial^2\dot v_x/(\partial\psi\,\partial u_L)
=-(F_{\max}/m)\sin\psi$.
The discrete derivatives used by the solver also include how heading and
velocity change within the RK4 step. Automatic differentiation is applied
through that complete step. A differential equation that is affine in its
controls need not yield a numerical transition whose second control
derivatives vanish.

The extra curvature costs computation and can be indefinite. DDP therefore
still needs regularization and nonlinear acceptance tests. Neither method
guarantees the globally best docking maneuver. For affine dynamics the
transition Hessians are zero, so the two backward constructions coincide
when all other choices agree.

## Thruster Limits and Stopping Criteria

How should the local solve account for a thruster that is already at maximum
force? The boat's controls must remain in $[-1,1]^2$. At zero state deviation,
the feedforward step is the solution of the box-constrained quadratic problem

$$
\min_k\ q_u^\top k+\frac12k^\top(H_{uu}+\mu I)k,
\qquad -\mathbf1-\bar{\mathbf u}_t\leq k\leq\mathbf1-\bar{\mathbf u}_t.
$$

For two controls, each coordinate is either free, fixed at its lower bound,
or fixed at its upper bound. The teaching implementation checks the nine
combinations, minimizes over each face, and keeps the best feasible candidate.
Positive-definite curvature makes this a strictly convex QP. Larger systems
can use an active-set solver rather than enumerate all combinations
{cite:p}`tassa2014control`.

The feedback rows of saturated controls are zero while their active set stays
fixed. For the free coordinates $F$, solve
$(H_{uu}+\mu I)_{FF}K_F=-(H_{ux})_F$.
Use the full substitution formulas to update the quadratic tail. With bounds,
this tail describes a neighborhood with the selected active sets; across
active-set changes the minimized expression is generally piecewise quadratic.
During the nonlinear trial, project the resulting control onto its box before
simulation. This enforces the physical bounds when a trial leaves the
neighborhood represented by the backward pass. The nonlinear cost check then
decides whether that trial should be retained.

The stopping test is separate from docking success. Let
$g=\nabla_{\mathbf U}J$ be the gradient of the original shooting objective,
computed by the adjoint recursion along the current nonlinear rollout.
With normalized controls, the solver stops when

$$
\left\|\mathbf U-\operatorname{clip}(\mathbf U-g,-1,1)\right\|_\infty
\leq10^{-5}.
$$

At an interior optimum this reduces to a small control gradient. At a bound
it also allows a gradient that points toward a forbidden improvement. The
experiment permits at most 120 accepted updates. If regularization exceeds
$10^{12}$ without an acceptable trial, the solver retains its last accepted
trajectory and reports failure to make progress. A small residual certifies
approximate first-order stationarity, not a globally optimal path or a
successful arrival.

(sec-docking-results)=
## Docking Trajectories and Recorded Futures

Do the two curvature choices produce the same maneuver from the same starting
plan? Both methods receive the coasting initialization, identical costs and
control limits, and the same stopping settings. The following tables report
their final nonlinear costs and arrival checks. Each final control sequence
is also replayed with four RK4 substeps per control interval, giving a
$0.025\ \mathrm s$ validation grid.

:::{include} artifacts/boat_docking/results.md
:::

All four runs satisfy the stopping test and the docking checks: position error
below $0.2\ \mathrm m$, heading error below $3^\circ$, speed below
$0.05\ \mathrm{m/s}$, angular speed below $1^\circ/\mathrm s$, bounded thrust,
and positive hull clearance. The maximum position difference between the
planning trajectory and the finer replay is below $10^{-6}\ \mathrm m$ in
these runs. Clearance is checked on the finer grid; this remains a sampled
validation, not a proof of continuous-time collision avoidance.

:::{figure} _static/boat_docking/paths.svg
:label: fig-boat-paths
:width: 100%

Final trajectories under the same initialization and objective. Solid blue
is iLQR and dashed orange is DDP; the dotted gray path is the initial coast.
Outlines show orientation every four seconds. DDP takes a different turning
maneuver in the angled approach. The two final paths nearly coincide in the
sideways-drift case.
:::

In the angled approach, iLQR reaches cost 8.0515 in 18 accepted updates, while
DDP reaches a different stationary trajectory with cost 11.9879 in 34 updates.
The DDP trajectory makes two full turns near the berth. Its final unwrapped
heading is approximately $720^\circ$, which the periodic heading cost treats
as the same orientation as zero. Reaching this stationary trajectory does
not remove the extra motion accumulated on the way there. In the
sideways-drift case, both reach cost 7.6674, with 13 accepted updates for iLQR
and 45 for DDP. These are outcomes of two particular local solves, not a
general ranking of the algorithms. The difference in the first case also
shows why comparing iteration counts alone can hide a difference in the
solutions obtained.

:::{figure} _static/boat_docking/convergence.svg
:label: fig-boat-convergence
:width: 100%

Actual nonlinear cost at every accepted iteration, including the initial
rollout at iteration zero. The vertical axis is logarithmic. Rejected trial
steps are excluded from the horizontal count; forward-evaluation and
backward-attempt counts are available in the downloadable diagnostics.
:::

Within one selected plan, simulation time has a different meaning from
optimizer iteration. At two seconds, for example, the boat occupies the
two-second state of that rollout. Its remaining states give a prediction of
where it will go if the rest of the stored controls are applied. Outlines
spaced equally in time spread apart while the boat is moving quickly and
cluster as it comes to rest. Reverse thrust helps remove momentum before
arrival.

:::{figure} _static/boat_docking/future-and-motion.svg
:label: fig-boat-future
:width: 100%

The final iLQR plan for the angled approach. The filled boat marks the selected
simulation time; translucent outlines show future poses every two seconds.
The right panel gives speed and both physical thrusts. Negative thrust during
the approach brakes the boat. The vertical marker identifies the same time
as the filled boat.
:::

:::{iframe} ../interactive/boat-docking.html
:width: 100%
:title: Boat docking with iLQR and DDP
:class: boat-docking-replay

Choose a scenario and optimizer iteration, then scrub or play simulation
time within that fixed plan. The path ahead and boat outlines display its
remaining predicted states. Arrival errors and plots belong to the selected
iteration, including intermediate plans that do not dock successfully.
:::

The replay reads recorded simulations, so exploring it does not rerun an
optimizer. Recomputing a plan from newly observed states is the additional
step introduced in [receding-horizon control](receding-horizon-control.md).

## Exercises

:::{exercise} Change the terminal weight
:label: ex-ilqr-two-step

In the two-step scalar example, replace the terminal cost by
$\tfrac\beta2(x_2-r)^2$, with $\beta>0$. Eliminate $u_1$ and then $u_0$.
Find both controls and the terminal state. What happens as $\beta\to\infty$?
:::

:::{solution} ex-ilqr-two-step
:class: dropdown

The last control is $u_1=\beta(r-x_1)/(1+\beta)$, and its minimized tail
is $\beta(x_1-r)^2/[2(1+\beta)]$. With $x_0=0$, minimizing the remaining
expression gives $u_0=\beta r/(1+2\beta)$, followed by the same value for
$u_1$. Thus $x_2=2\beta r/(1+2\beta)$. As the terminal weight grows, the
two controls tend to $r/2$ and the terminal state tends to $r$.
:::

:::{exercise} The linear-quadratic special case
:label: ex-ilqr-linear-case

Suppose the dynamics are affine and every cost is quadratic, with a strictly
convex reduced objective and no control bounds. Explain why one unregularized
backward pass and a full nonlinear rollout solve the problem from any
feasible initial trajectory. What changes when using DDP?
:::

:::{solution} ex-ilqr-linear-case
:class: dropdown

The local expansions are exact. Backward elimination therefore minimizes the
original quadratic problem in perturbation coordinates, and $\alpha=1$
recovers its solution. The affine simulator agrees exactly with the
linearized perturbation dynamics. All transition Hessians vanish, so DDP
produces the same correction. This is also a useful implementation test:
compare the returned controls with a dense solve of the reduced quadratic
objective.
:::

:::{exercise} A curvature term in the boat model
:label: ex-ilqr-boat-curvature

Temporarily discretize the boat with forward Euler. Derive the mixed derivative
$\partial^2 f^{v_x}/(\partial\psi\,\partial u_L)$. Identify where it enters
the DDP backward pass and why iLQR omits it.
:::

:::{solution} ex-ilqr-boat-curvature
:class: dropdown

The Euler velocity update contains
$h(F_{\max}/m)(u_L+u_R)\cos\psi$. The mixed derivative is therefore
$-h(F_{\max}/m)\sin\psi$. DDP multiplies it by the $v_x$ component of
$p_{t+1}$ and adds it to the corresponding state--control curvature entry.
iLQR uses the first-order transition, whose second derivatives are zero
within the local model. The actual experiment differentiates RK4 instead,
so its mixed derivative includes the intermediate integration stages.
:::

:::{exercise} Remove the stopping penalty
:label: ex-ilqr-terminal-speed

Set the terminal translational-velocity weight to zero and rerun the angled
approach. Compare position error, terminal speed, and controls with the
default run. Repeat with an eight-second horizon. Does arriving near the
berth imply that docking succeeded?
:::

:::{solution} ex-ilqr-terminal-speed
:class: dropdown

Use `make_problem(velocity_weight=0.)` and keep every other choice fixed for
the first comparison. For the shorter experiment, set `steps=80` in
`BoatParameters` and regenerate the initial controls. Removing the terminal
velocity term removes an explicit incentive to stop at the final time.
Running position and velocity costs can still lead to a slow arrival,
especially over the longer horizon, so failure is not inevitable. Measure
the final speed independently of position error and retain the same docking
tolerances for both objectives. A low-cost pass through the target is not
a successful stationary arrival.
:::

:::{exercise} Saturation and a rejected step
:label: ex-ilqr-saturation

For a positive-definite $2\times2$ matrix $H$ with nonzero off-diagonal entries,
compare the exact box-QP solution with clipping $-H^{-1}q$ componentwise.
Explain why they can differ. What should the trajectory optimizer retain if
every nonlinear trial is rejected?
:::

:::{solution} ex-ilqr-saturation
:class: dropdown

Fixing one coordinate at a bound changes the stationarity equation for the
other coordinate through the off-diagonal entry of $H$. Componentwise
clipping does not solve that new equation. The face minimization does.
If no nonlinear trial is accepted, the solver retains its previous controls
and states and reports that it could not make progress; it must not replace
them with an unaccepted trial or report convergence from rejection alone.
:::

## Computational Sources

The implementation separates the generic backward recursion from the boat
model and the artifact builder:

- {download}`Shared iLQR and DDP solver <code/trajectory_optimization.py>`
- {download}`Boat dynamics, costs, and validation <code/boat_docking.py>`
- {download}`Experiment and figure builder <scripts/build_boat_docking_artifacts.py>`
- {download}`Numerical diagnostics and source hashes <artifacts/boat_docking/metrics.json>`

Run `uv run python scripts/build_boat_docking_artifacts.py` from the repository
root to reproduce the solves, figures, results table, and browser data. Normal
book builds read those artifacts. Tests independently check the derivatives,
the composed DDP curvature, the scalar calculation, and agreement between
backward elimination and a dense quadratic solve.

## Summary and Outlook

A feasible nonlinear rollout supplies the point around which dynamics and
costs are approximated. Backward elimination of a local quadratic problem
then supplies both a feedforward change and a state-dependent correction.
iLQR repeats that calculation with linearized dynamics; DDP includes the
second-order chain-rule terms from the transition. Nonlinear rollouts,
regularization, and acceptance tests connect these local calculations to a
decreasing sequence of actual trajectory costs.

The docking experiments use a discrete transition obtained by integrating
an ordinary differential equation. [Continuous-time transcription and
collocation](continuous-time-collocation.md) develops other ways to represent
continuous trajectories inside a finite optimization problem. Later,
[finite-horizon dynamic programming](finite-horizon-dp.md) gives a broader
interpretation of the quadratic tail: in the linear-quadratic case, the
function produced by elimination is the optimal cost-to-go.
