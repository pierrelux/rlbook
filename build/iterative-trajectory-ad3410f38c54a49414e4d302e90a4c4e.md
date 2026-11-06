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
**DDP**. Both methods compute corrections around the current plan. The
local problem is the [linear-quadratic regulator](#sec-lqr-dp) solved by dynamic
programming in the previous chapter, and its backward recursion can also be
read as substitution and quadratic minimization.

The chapter is a bridge between the exact dynamic programming of the previous
chapter and the [stochastic dynamic programming](stochastic-dp.md) of the
next. iLQR reuses the LQR backward recursion, but only locally: its quadratic
models and its feedback gains describe a neighborhood of one trajectory, not
the whole state space.

## Learning Goals

After working through the derivations and examples, you should be able to:

- Construct a local quadratic optimal-control problem around a feasible rollout.
- Eliminate its controls backward and recover an affine correction at each step.
- Implement nonlinear trial rollouts with line search and regularization.
- Identify the dynamics-curvature terms that distinguish DDP from iLQR.
- Assess a docking plan using its cost, arrival velocity, and hull clearance.
- Explain how thermal load and nonlinear power loss shape a bounded cooling schedule.

## Prerequisites

The chapter uses the [finite-horizon optimal-control formulation](discrete-time-optimal-control.md),
[single shooting](numerical-trajectory-optimization.md), the
[linear-quadratic regulator](#sec-lqr-dp) from finite-horizon dynamic
programming, and first and second derivatives. The
[SQP section of the nonlinear-programming appendix](#sec-sqp-approach)
introduces general SQP subproblems, and the [initial-value-problem appendix](appendix_ivps.md) reviews
numerical integration.

(sec-boat-docking)=
## A Docking Problem

Docking requires the boat to reach the berth, align with the quay, and shed
its momentum. If the boat reaches the right position with appreciable
translational or angular velocity, its momentum will carry it away again.

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
$-0.5\ \mathrm{m/s}$. Angles are in radians. Both experiments use $T-1=200$
piecewise-constant controls, each held for $h=0.1\ \mathrm s$, for a total
of 20 seconds. Thus $\mathbf x_1$ is the initial state and $\mathbf x_T$ is the
arrival state, as in the preceding chapters. One fourth-order Runge--Kutta
step gives $\mathbf x_{t+1}=\mathbf f_t(\mathbf x_t,\mathbf u_t)$ for
$t=1,\ldots,T-1$.

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
c_T(\mathbf x)
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
## Taylor Models of Dynamics and Cost

The local model describes the change in cost caused by a small change in the
thrust sequence. Start from the finite-horizon problem

$$
\begin{aligned}
\min_{\mathbf U}\quad &J(\mathbf U)
 :=c_T(\mathbf x_T)+\sum_{t=1}^{T-1}c_t(\mathbf x_t,\mathbf u_t),\\
\text{s.t.}\quad &\mathbf x_{t+1}=\mathbf f_t(\mathbf x_t,\mathbf u_t),
\quad t=1,\ldots,T-1,\\
&\mathbf x_1=\mathbf x_{\mathrm{init}}.
\end{aligned}
$$

Here $\mathbf U=(\mathbf u_1,\ldots,\mathbf u_{T-1})$ is the complete control
sequence. Its rollout determines every later state. This is an open-loop
problem: the decision variable is a sequence of commands fixed in advance, not
a rule that reacts to the state. iLQR returns such a sequence. The feedback
gains $K_t$ that appear below are a by-product of the computation. Near the
returned plan they tell how to adjust each command when the state drifts from
the planned one, through $\mathbf u=\bar{\mathbf u}+K(\mathbf x-\bar{\mathbf x})$,
and they say nothing about states far from it.

{numref}`fig-boat-tube` shows what "near the plan" means. The boat follows the
final docking plan and receives a small push after two seconds. Replaying the
planned thrusts ignores the push. The boats drift calmly but none reaches the
berth: they miss it by 0.9 m on average and by up to 2.2 m. Applying the local
feedback steers all twelve pushed runs back onto the plan, and they arrive
within a millimetre of it. These runs stay inside a thin tube around the
planned trajectory.

The tube has a limited width. With pushes twice as large, shown in the
animation below, several feedback runs leave it and two never reach the
berth. Their corrections ask for more thrust than the thrusters can give, and
the boat moves far enough from the plan that the linear model is no longer
accurate. The gains describe
a neighborhood of one trajectory, and this whole chapter works inside such
neighborhoods.

:::{figure} _static/boat_docking/tube.svg
:label: fig-boat-tube
:width: 100%

Twelve random pushes at 2 s (velocity changes up to 0.3 m/s in each direction,
with small heading and spin changes) applied to the final iLQR plan for the
angled approach. Left: the planned thrusts are replayed unchanged. Right: the
thrusts follow the local feedback law, clipped to the thruster limits. The
shaded tube, drawn in both panels, is the envelope of the feedback runs.
Circles mark the positions at 20 s.
:::

:::{iframe} ../interactive/boat-tube.html
:width: 100%
:title: Pushed boats with and without local feedback
:class: boat-tube-replay

Play or scrub simulation time, and switch to the larger pushes to see boats
leave the tube. The counters report how many boats are inside the tube and at
the berth at the selected time.
:::

We cannot solve this nonlinear problem directly. The previous chapter showed
that a linear-quadratic regulator problem can be solved exactly by a backward
recursion. The idea of this chapter is to build an LQR problem that resembles
ours near a current plan, solve it, use its solution to improve the plan, and
repeat.

Set aside the control bounds while deriving this local problem. The **current
plan** is a control sequence $\bar{\mathbf U}$ together with the states
obtained by rolling it out,
$\bar{\mathbf x}_{t+1}=\mathbf f_t(\bar{\mathbf x}_t,\bar{\mathbf u}_t)$.
For the boat, the first plan is the zero-thrust coast. The plan is our current
guess, not a reference that the boat must track. To ask what happens if we
change it, consider nudges $\delta\mathbf x_t=\mathbf x_t-\bar{\mathbf x}_t$
and $\delta\mathbf u_t=\mathbf u_t-\bar{\mathbf u}_t$, stacked as
$\delta\mathbf z_t=(\delta\mathbf x_t,\delta\mathbf u_t)$. These are
hypothetical changes that we are free to choose, not measured errors.

The linear model of the transition follows the usual Taylor steps. A scalar
function of one variable satisfies $f(a+h)\approx f(a)+f'(a)h$. With two
inputs the partial derivatives add:
$f(\bar x+\delta x,\bar u+\delta u)\approx f(\bar x,\bar u)
+\partial_x f\,\delta x+\partial_u f\,\delta u$. The transition $\mathbf f_t$
has one such expansion for each of its $n_x$ components. Each component gives
one row, and stacking the rows gives the Jacobian matrices $A_t$ and $B_t$:

$$
\begin{aligned}
\mathbf f_t(\bar{\mathbf x}_t+\delta\mathbf x_t,
             \bar{\mathbf u}_t+\delta\mathbf u_t)
&=\mathbf f_t(\bar{\mathbf x}_t,\bar{\mathbf u}_t)
  +A_t\delta\mathbf x_t+B_t\delta\mathbf u_t
  +O(\|\delta\mathbf z_t\|^2),\\
A_t&=\left.\frac{\partial\mathbf f_t}{\partial\mathbf x}
      \right|_{(\bar{\mathbf x}_t,\bar{\mathbf u}_t)},\\
B_t&=\left.\frac{\partial\mathbf f_t}{\partial\mathbf u}
      \right|_{(\bar{\mathbf x}_t,\bar{\mathbf u}_t)}.
\end{aligned}
$$

The left side is the transition evaluated at the nudged state and control,
which is the new next state $\mathbf x_{t+1}$. Because the current plan is a
rollout, the first term on the right is
$\mathbf f_t(\bar{\mathbf x}_t,\bar{\mathbf u}_t)=\bar{\mathbf x}_{t+1}$.
Subtracting $\bar{\mathbf x}_{t+1}$ from both sides turns the left side into
$\delta\mathbf x_{t+1}$ and cancels that term on the right. Dropping the
quadratic remainder gives the linearized dynamics
$\delta\mathbf x_{t+1}=A_t\delta\mathbf x_t+B_t\delta\mathbf u_t$.
Both rollouts start from the same state, so $\delta\mathbf x_1=\mathbf0$.
There is no constant defect because the current plan satisfies the discrete
dynamics. Independently chosen state guesses would generally leave such a
defect. We write $F_t=[A_t\ B_t]$ for the combined Jacobian.

The running cost is expanded by the same steps, one order higher. In one
variable, $f(a+h)\approx f(a)+f'(a)h+\tfrac12f''(a)h^2$. Applied to the
stacked vector $\delta\mathbf z_t$, the derivative becomes a gradient and the
second derivative becomes a Hessian matrix. Splitting $\delta\mathbf z_t$ back
into its state and control parts cuts the gradient into $c_{x,t}$ and
$c_{u,t}$, and cuts the Hessian into four blocks: $c_{xx,t}$ for curvature in
the state, $c_{uu,t}$ for curvature in the control, and the mixed blocks
$c_{xu,t}$ and $c_{ux,t}=c_{xu,t}^\top$. The quadratic term therefore reads
$\tfrac12\delta\mathbf x_t^\top c_{xx,t}\delta\mathbf x_t
+\delta\mathbf u_t^\top c_{ux,t}\delta\mathbf x_t
+\tfrac12\delta\mathbf u_t^\top c_{uu,t}\delta\mathbf u_t$, where the two
mixed blocks have combined into one term. In block-matrix form,

$$
c_t(\bar{\mathbf x}_t+\delta\mathbf x_t,
    \bar{\mathbf u}_t+\delta\mathbf u_t)
 = \bar c_t+
\begin{bmatrix}c_{x,t}\\c_{u,t}\end{bmatrix}^{\!\top}\delta\mathbf z_t
+\frac12\delta\mathbf z_t^\top
\begin{bmatrix}c_{xx,t}&c_{xu,t}\\c_{ux,t}&c_{uu,t}\end{bmatrix}
\delta\mathbf z_t+o(\|\delta\mathbf z_t\|^2).
$$

Here $\bar c_t=c_t(\bar{\mathbf x}_t,\bar{\mathbf u}_t)$ and
$c_{x,t}=\nabla_{\mathbf x}c_t(\bar{\mathbf x}_t,\bar{\mathbf u}_t)$;
the other gradient and Hessian blocks are evaluated at the same pair from the
current plan.
We denote the stacked gradient by $c_{z,t}$ and the displayed block Hessian by
$c_{zz,t}$. Gradients are column vectors, matching the convention of the
preceding chapters. We retain the linear terms: individual stage gradients
need not vanish even at an optimal trajectory because the dynamics couple
stages, and at a non-optimal plan they also drive its correction.
The $c_{xu,t}$ block measures how the marginal cost of a control changes with
the state.
Even if the running cost has no such cross term, eliminating later controls
can create one in the quadratic tail.

The terminal cost has the same expansion, with no control coordinate:

$$
\begin{aligned}
c_T(\bar{\mathbf x}_T+\delta\mathbf x_T)
&=c_T(\bar{\mathbf x}_T)+p_T^\top\delta\mathbf x_T\\
&\quad+\frac12\delta\mathbf x_T^\top P_T\delta\mathbf x_T
 +o(\|\delta\mathbf x_T\|^2),\\
p_T&=\nabla_{\mathbf x}c_T(\bar{\mathbf x}_T),\qquad
P_T=\nabla^2_{\mathbf x\mathbf x}c_T(\bar{\mathbf x}_T).
\end{aligned}
$$

Dropping the remainders in these three Taylor expansions gives the local
problem. Up to the constant $\sum_t\bar c_t+c_T(\bar{\mathbf x}_T)$, which does
not depend on the nudges, it reads

$$
\begin{aligned}
\min_{\delta\mathbf u_1,\ldots,\delta\mathbf u_{T-1}}\quad
&\sum_{t=1}^{T-1}\Big(c_{z,t}^\top\delta\mathbf z_t
  +\frac12\delta\mathbf z_t^\top c_{zz,t}\,\delta\mathbf z_t\Big)
 +p_T^\top\delta\mathbf x_T
 +\frac12\delta\mathbf x_T^\top P_T\,\delta\mathbf x_T,\\
\text{s.t.}\quad
&\delta\mathbf x_{t+1}=A_t\delta\mathbf x_t+B_t\delta\mathbf u_t,
 \quad t=1,\ldots,T-1,\\
&\delta\mathbf x_1=\mathbf0.
\end{aligned}
$$ (eq-ilqr-local-problem)

This is an LQR problem of the kind solved in the previous chapter: linear
dynamics in the nudges, and a cost that is quadratic in them. Compared with
the standard LQR problem it has linear cost terms and a state--control cross
term, which the next section handles. Solving it gives a proposed change to
the plan.

Notice what is missing from this local problem. Only the costs were expanded
to second order. The dynamics were expanded to first order, so their
curvature appears nowhere in {eq}`eq-ilqr-local-problem`. A general
[SQP method](#sec-sqp-approach) also forms a local quadratic subproblem, but
not quite this one. SQP takes a quadratic model of the Lagrangian and
linearizes the constraints. Here the constraints are the dynamics, so the
Hessian of the Lagrangian is the cost Hessian plus the sum over $i$ of the
multiplier $\lambda_i$ times the Hessian of the $i$th component of the
transition. The shooting formulation is unconstrained
only in the controls, because the simulator enforces the dynamics, but the
local problem above still has the linearized dynamics as constraints on the
nudges. iLQR is therefore SQP with the multiplier-weighted dynamics Hessians
dropped. Those terms return in the [DDP section](#sec-ddp-curvature), where
the coefficient $p_{t+1}$ of the eliminated tail plays the role of the
multiplier, the costate.

## Backward Elimination

The local problem {eq}`eq-ilqr-local-problem` is a deterministic
finite-horizon problem with linear dynamics and quadratic costs. It is an LQR
problem, so the backward induction of the previous chapter solves it. This
section recalls that algorithm, specializes it to LQR, and then applies the
same specialization to the local problem. Each backup minimizes over one
control and removes it from the problem, which is why the backward pass of
iLQR is often called backward elimination.

### Backward induction

For a deterministic problem with dynamics $\mathbf x_{t+1}=\mathbf f_t(\mathbf x_t,\mathbf u_t)$,
stage costs $c_t$ and terminal cost $c_T$,
{prf:ref}`backward-recursion-deterministic` computes the optimal cost-to-go
$J_t^\star$ backward in time:

$$
\begin{aligned}
J_T^\star(\mathbf x)&=c_T(\mathbf x),\\
J_t^\star(\mathbf x)&=\min_{\mathbf u}\Big[c_t(\mathbf x,\mathbf u)
  +J_{t+1}^\star\big(\mathbf f_t(\mathbf x,\mathbf u)\big)\Big],
  \qquad t=T-1,\ldots,1.
\end{aligned}
$$

The minimizer at each state defines the policy $\pi_t^\star(\mathbf x)$. A
forward pass from the initial state then applies $\mathbf u_t=\pi_t^\star(\mathbf x_t)$
and recovers the optimal trajectory. In general $J_t^\star$ is a function on
the whole state space, which the previous chapter stored on a grid.

### The LQR case: the Riccati recursion

With linear dynamics $\mathbf x_{t+1}=A_t\mathbf x_t+B_t\mathbf u_t$, stage
cost $\tfrac12\mathbf x^\top Q_t\mathbf x+\tfrac12\mathbf u^\top R_t\mathbf u$
and terminal cost $\tfrac12\mathbf x^\top Q_T\mathbf x$, every cost-to-go is
quadratic, $J_t^\star(\mathbf x)=\tfrac12\mathbf x^\top P_t\mathbf x$, so the
backup only has to update a matrix. Substituting the dynamics into
$J_{t+1}^\star$ makes the bracket a quadratic function of the stacked state and
control,

$$
\frac12
\begin{bmatrix}\mathbf x\\\mathbf u\end{bmatrix}^{\!\top}
\begin{bmatrix}
Q_t+A_t^\top P_{t+1}A_t & A_t^\top P_{t+1}B_t\\
B_t^\top P_{t+1}A_t & R_t+B_t^\top P_{t+1}B_t
\end{bmatrix}
\begin{bmatrix}\mathbf x\\\mathbf u\end{bmatrix}.
$$

Setting its control gradient to zero gives the linear feedback
$\mathbf u_t^\star=-K_t\mathbf x$, and substituting it back gives the
**Riccati recursion** of {prf:ref}`backward-recursion-lqr`:

$$
\begin{aligned}
K_t&=(R_t+B_t^\top P_{t+1}B_t)^{-1}B_t^\top P_{t+1}A_t,\\
P_t&=Q_t+A_t^\top P_{t+1}A_t
 -A_t^\top P_{t+1}B_t(R_t+B_t^\top P_{t+1}B_t)^{-1}B_t^\top P_{t+1}A_t,
\end{aligned}
$$

starting from $P_T=Q_T$.

### Riccati for the local problem

The local problem has the same structure, with the nudges
$\delta\mathbf x_t,\delta\mathbf u_t$ as state and control. Its terminal
cost-to-go is the terminal Taylor model, and its backup is

$$
S_t(\delta\mathbf x)=\min_{\delta\mathbf u}\Big[
c_{z,t}^\top\delta\mathbf z+\frac12\delta\mathbf z^\top c_{zz,t}\,\delta\mathbf z
+S_{t+1}(A_t\delta\mathbf x+B_t\delta\mathbf u)\Big],
\qquad \delta\mathbf z=(\delta\mathbf x,\delta\mathbf u).
$$

Two features differ from textbook LQR. The current plan is generally not
optimal, so the local costs have linear terms, the gradients $c_{z,t}$ and
$p_T$. The stage cost may also contain a state--control cross term
$c_{ux,t}$. Because of the linear terms, the cost-to-go needs a linear part as
well:

$$
S_{t+1}(\delta\mathbf x)=s_{t+1}+p_{t+1}^\top\delta\mathbf x
+\frac12\delta\mathbf x^\top P_{t+1}\delta\mathbf x,
$$

with $p_T$ and $P_T$ the gradient and Hessian of $c_T$ at the end of the
plan. Substituting the linear dynamics into $S_{t+1}$ makes the bracket a
quadratic in $\delta\mathbf z$ with linear coefficient $q_t$ and Hessian
$M_t$. Written in blocks, next to the LQR matrix above,

$$
\begin{aligned}
q_{x,t}&=c_{x,t}+A_t^\top p_{t+1}, &
M_{xx,t}&=c_{xx,t}+A_t^\top P_{t+1}A_t,\\
q_{u,t}&=c_{u,t}+B_t^\top p_{t+1}, &
M_{ux,t}&=c_{ux,t}+B_t^\top P_{t+1}A_t,\\
& & M_{uu,t}&=c_{uu,t}+B_t^\top P_{t+1}B_t.
\end{aligned}
$$

The Hessian blocks are the LQR blocks with $Q_t$ replaced by $c_{xx,t}$,
$R_t$ by $c_{uu,t}$, and the cross term $c_{ux,t}$ added. The vector $q_t$ is
new. In stacked form, with $F_t=[A_t\ B_t]$, these are
$q_t=c_{z,t}+F_t^\top p_{t+1}$ and $M_t=c_{zz,t}+F_t^\top P_{t+1}F_t$. We use
$M_t$ for this Hessian because $H_t$ denoted the Hamiltonian in the
Pontryagin chapter.

Setting the control gradient to zero, $M_{uu,t}\delta\mathbf u+M_{ux,t}\delta\mathbf x+q_{u,t}=0$,
gives an affine law instead of a linear one:

$$
\delta\mathbf u_t=k_t+K_t\delta\mathbf x_t,\qquad
k_t=-M_{uu,t}^{-1}q_{u,t},\qquad K_t=-M_{uu,t}^{-1}M_{ux,t}.
$$

The feedback gain $K_t$ plays the role of the LQR gain, written here with the
sign included. The feedforward term $k_t$ comes from the linear terms: it
changes the plan even when the state nudge is zero. Substituting the control
back gives the coefficients for the preceding step,

$$
\begin{aligned}
p_t&=q_{x,t}-M_{ux,t}^\top M_{uu,t}^{-1}q_{u,t},\\
P_t&=M_{xx,t}-M_{ux,t}^\top M_{uu,t}^{-1}M_{ux,t}.
\end{aligned}
$$

The update of $P_t$ is the Riccati recursion. Written out, it reads
$P_t=c_{xx,t}+A_t^\top P_{t+1}A_t-(c_{ux,t}+B_t^\top P_{t+1}A_t)^\top
(c_{uu,t}+B_t^\top P_{t+1}B_t)^{-1}(c_{ux,t}+B_t^\top P_{t+1}A_t)$, which is
the LQR formula when $c_{ux,t}=0$. The update of $p_t$ is the new part. It
carries the cost gradients backward, and with zero gradients it stays zero
and $k_t$ vanishes. In code, these formulas are linear solves with
$M_{uu,t}$; an explicit matrix inverse is unnecessary.

````{prf:algorithm} Backward pass for the local problem
:label: alg-ilqr-backward-pass

**Input:** Jacobians $A_t,B_t$ and cost derivatives $c_{z,t},c_{zz,t}$ along the current plan; terminal gradient $p_T$ and Hessian $P_T$.

**Output:** Feedforward terms $k_t$ and gains $K_t$ for $t=1,\ldots,T-1$.

1. For $t=T-1,\ldots,1$:
   1. Form $q_t=c_{z,t}+F_t^\top p_{t+1}$ and $M_t=c_{zz,t}+F_t^\top P_{t+1}F_t$.
   2. Solve $k_t=-M_{uu,t}^{-1}q_{u,t}$ and $K_t=-M_{uu,t}^{-1}M_{ux,t}$.
   3. Update $p_t=q_{x,t}-M_{ux,t}^\top M_{uu,t}^{-1}q_{u,t}$ and $P_t=M_{xx,t}-M_{ux,t}^\top M_{uu,t}^{-1}M_{ux,t}$.
2. Return $k_t$ and $K_t$. A forward pass from $\delta\mathbf x_1=\mathbf0$ with $\delta\mathbf u_t=k_t+K_t\delta\mathbf x_t$ and $\delta\mathbf x_{t+1}=A_t\delta\mathbf x_t+B_t\delta\mathbf u_t$ solves the local problem.
````

The tail $S_t$ is the cost-to-go of the local problem only. It is accurate
near the current plan, and it is not a value function of the nonlinear
system over its whole state space. Unlike the grid of the previous chapter, it
is stored as a vector and a matrix per step, so the backward pass costs a
fixed amount of work per step and grows linearly with the horizon. For linear
dynamics and quadratic costs it solves the original problem exactly whenever
each $M_{uu,t}$ is positive definite. Along a nonlinear trajectory, $A_t$,
$B_t$, and the cost derivatives describe only a local approximation. They
must be recomputed after the plan changes. {ref}`ex-ilqr-two-step` carries
out one backward pass by hand on a scalar problem.

(sec-ilqr-rollout)=
## Nonlinear Rollouts and iLQR

The backward pass returns, at every step, an affine law
$\delta\mathbf u_t=k_t+K_t\delta\mathbf x_t$ for the local problem. iLQR turns
it into a new plan and repeats:

1. **Roll out.** Simulate the current controls $\bar{\mathbf U}$ through the
   true dynamics. This gives the current plan $\bar{\mathbf x}_t$,
   $\bar{\mathbf u}_t$ and its cost.
2. **Backward pass.** Linearize the dynamics and expand the costs around this
   plan, then solve the local LQR problem with
   {prf:ref}`alg-ilqr-backward-pass`. This gives $k_t$ and $K_t$.
3. **Forward pass.** Run the affine law in closed loop through the true
   simulator, starting from the same initial state:

   $$
   \begin{aligned}
   \mathbf x_1^+&=\mathbf x_{\mathrm{init}},\\
   \mathbf u_t^+&=\bar{\mathbf u}_t+k_t+K_t(\mathbf x_t^+-\bar{\mathbf x}_t),\\
   \mathbf x_{t+1}^+&=\mathbf f_t(\mathbf x_t^+,\mathbf u_t^+).
   \end{aligned}
   $$

   The new states and controls are the next plan.
4. **Repeat** until the plan stops changing.

This loop is the **iterative linear-quadratic regulator**, iLQR
{cite:p}`li2004iterative`: an LQR problem is solved at every iteration, around
a different trajectory. The control law in step 3 is the local solution
$\mathbf u=\bar{\mathbf u}+\delta\mathbf u$, with the nudge
$\delta\mathbf x_t$ measured as the simulated state minus the planned one.
The feedback term matters because the true simulator does not follow the
linear model exactly. When the new states drift from the linear prediction,
$K_t$ adjusts the controls the way the local model says the best control
changes with the state.

The loop always linearizes around a trajectory that the true dynamics
produced. The gains from the old linearization generate the new trajectory,
and that trajectory gives the next linearization. Everything in steps 2 and 3
follows from the derivation above: $q_t$, $M_t$, $k_t$, $K_t$, $p_t$ and
$P_t$ all come out of the LQR algebra. Two safeguards are added by hand, a
step size and a regularization, and the rest of this section explains why
each is needed.

### Step size and line search

The local model is accurate only near the current plan. Far from the optimum,
the full step of the forward pass can move the boat well outside that
neighborhood, and the true cost can then go up instead of down. Nothing in the
Riccati derivation prevents this.

The usual fix comes from numerical optimization. Treat the LQR solution as a
search direction, as one would treat a Newton step, and decide separately how
far to move along it. This is a damped Newton method. The step size $\alpha$
is not part of the Riccati derivation; it is added afterward and multiplies
only the feedforward term:

$$
\mathbf u_t^+=\bar{\mathbf u}_t+\alpha k_t+K_t(\mathbf x_t^+-\bar{\mathbf x}_t).
$$

Putting $\alpha$ on $k_t$ alone is a choice, and the reason is that it gives a
path between two known trajectories. At $\alpha=0$ every state nudge stays
zero, by induction through the forward pass, and the old plan comes back. At
$\alpha=1$ the forward pass takes the full LQR step, which is the exact
solution when the problem is linear-quadratic. In between, the gains keep
each trial consistent with the true dynamics.

A **line search** chooses $\alpha$ at every iteration. Try the full step
$\alpha=1$, simulate, and keep the result if the true cost went down enough.
Otherwise halve $\alpha$ and simulate again. Each trial is one run through the
nonlinear simulator. The step size is therefore not a fixed hyperparameter;
it adapts at every iteration. When the linear model is accurate, as near
convergence, the full step usually passes at once. The real settings are the
halving schedule and the meaning of "enough".

The implementation tries $1,1/2,\ldots,2^{-11}$. For "enough", the backward
pass predicts the reduction of the local model. With

$$
d_1=\sum_{t=1}^{T-1}q_{u,t}^\top k_t,\qquad
d_2=\frac12\sum_{t=1}^{T-1}k_t^\top M_{uu,t}k_t,
$$

the predicted reduction is
$\widehat\Delta J(\alpha)=-\alpha d_1-\alpha^2d_2$.
A trial is accepted when its true cost decreases strictly and by at least a
small fraction of this prediction,
$J(\bar{\mathbf U})-J(\mathbf U^+)\geq10^{-4}\max(\widehat\Delta J(\alpha),0)$.
The prediction guides acceptance; it does not replace evaluation of the
original objective.

### Regularization

The second problem appears in the backward pass itself. Away from the
optimum, the control curvature $M_{uu,t}$ may be indefinite or nearly
singular, and then the control solve fails or proposes a huge change. Replace
the matrix used in the control solve by $M_{uu}+\mu I$, where $\mu>0$.
Increasing $\mu$ makes the correction more conservative and can make the solve
positive definite. The code checks a Cholesky factorization, increases $\mu$
after a failed backward pass or line search, and decreases it after an
accepted step. The numerical defaults start at $10^{-4}$, multiply by ten
after failure, and divide by three after acceptance, with a floor of
$10^{-9}$.

When regularization changes $k$ and $K$, the formulas for $p_t$ and $P_t$
above no longer describe substitution into the original quadratic model. The
implementation therefore uses the full expressions

$$
\begin{aligned}
p_t&=q_x+M_{ux}^\top k+K^\top q_u+K^\top M_{uu}k,\\
P_t&=M_{xx}+M_{ux}^\top K+K^\top M_{ux}+K^\top M_{uu}K.
\end{aligned}
$$

Here $M_{uu}$ is the original curvature block; regularization was used to
choose the correction. These expressions also remain applicable when some
controls are fixed at their bounds.

With both safeguards in place, the complete method reads as follows.

````{prf:algorithm} iLQR
:label: alg-trajectory-ilqr

**Input:** Discrete transition, running and terminal costs, initial state, and a feasible initial control sequence.

1. Roll out the current controls and evaluate their nonlinear cost.
2. Differentiate the transition and costs along this rollout.
3. Run the backward pass ({prf:ref}`alg-ilqr-backward-pass`) to obtain $k_t$ and $K_t$. Increase the regularization $\mu$ and retry if a control solve fails.
4. Run forward passes with $\alpha=1,\tfrac12,\tfrac14,\ldots$ through the nonlinear simulator.
5. Accept the first trial whose true cost decreases sufficiently, and decrease $\mu$. If none does, increase $\mu$ and return to step 3.
6. Repeat from step 2 until the control-gradient residual is small or a declared iteration or regularization limit is reached.

**Output:** The last accepted trajectory, its controls and local gains, an iteration history, and an explicit termination status.
````

The iteration uses second derivatives of the costs but only first derivatives
of the dynamics. It is often described as a Gauss--Newton approximation to
DDP. That terminology should not obscure what is approximated: the omitted
terms come from dynamics curvature, as the next section shows. If a nonlinear
cost is itself a sum of squared residuals, replacing its Hessian by a
residual-Jacobian product would be a further approximation; the docking code
uses the exact cost Hessian.

The returned gains describe a neighborhood of the final plan.
The docking animation below executes one stored plan. The gains used inside
the optimizer do not imply that the animation replans as the boat moves.

(sec-ddp-curvature)=
## Dynamics Curvature and DDP

iLQR drops the quadratic remainder in the transition's Taylor expansion. For
a turning boat, a change in heading changes the direction of thrust, so
products of state and control changes can affect the next state. DDP keeps
these second-order terms in the backward pass. Everything else stays the
same.

### The chain rule in one variable

Each backup minimizes a stage cost plus the cost-to-go evaluated at the next
state. With one variable, write $g$ for the cost-to-go and $f$ for the
dynamics. The second derivative of the composition has two terms:

$$
(g\circ f)''=g''\,(f')^2+g'\,f''.
$$

The first term is the curvature of the cost-to-go times the squared slope of
the dynamics. The second is the slope of the cost-to-go times the curvature of
the dynamics. iLQR keeps the first term only. DDP keeps both.

### One backup, expanded to second order

Write the quantity minimized in one backup as a function of the stacked nudge
$\delta\mathbf z_t=(\delta\mathbf x_t,\delta\mathbf u_t)$:

$$
c_t(\bar{\mathbf z}_t+\delta\mathbf z_t)
+S_{t+1}\big(\mathbf f_t(\bar{\mathbf z}_t+\delta\mathbf z_t)-\bar{\mathbf x}_{t+1}\big).
$$

Expand it to second order in $\delta\mathbf z_t$. The next-state nudge has a
linear part $F_t\delta\mathbf z_t$ and, for component $i$, a quadratic part
$\tfrac12\delta\mathbf z_t^\top f^i_{zz,t}\delta\mathbf z_t$, where
$f^i_{zz,t}$ is the Hessian of the $i$th component of $\mathbf f_t$ along the
current plan. Inserted in the tail, the quadratic part of $S_{t+1}$ sees only
the linear part; the rest is of order three or higher. The linear part
$p_{t+1}^\top\delta\mathbf x_{t+1}$ of the tail, however, sees both. The
linear coefficient is the same $q_t=c_{z,t}+F_t^\top p_{t+1}$ as before, and
the quadratic coefficient is

$$
M_t=c_{zz,t}
+\underbrace{F_t^\top P_{t+1}F_t}_{g''(f')^2}
+\underbrace{\sum_{i=1}^{n_x}p_{t+1,i}\,f^i_{zz,t}}_{g'f''}.
$$

iLQR drops the last term:

$$
M_t^{\mathrm{iLQR}}=c_{zz,t}+F_t^\top P_{t+1}F_t
+\cancel{\textstyle\sum_i p_{t+1,i}\,f^i_{zz,t}}.
$$

These are the multiplier-weighted dynamics Hessians that the
[local problem](#sec-local-quadratic-trajectory) left out. The tail gradient
$p_{t+1}$ plays the role of the SQP multiplier: it is the costate, the
sensitivity of the remaining cost to the next state.

### DDP keeps the term

DDP {cite:p}`tassa2014control` keeps the dynamics-curvature term. In blocks,

$$
\begin{aligned}
M_{xx,t}^{\mathrm{DDP}}&=c_{xx,t}+A_t^\top P_{t+1}A_t
                    +\sum_i p_{t+1,i} f^i_{xx,t},\\
M_{ux,t}^{\mathrm{DDP}}&=c_{ux,t}+B_t^\top P_{t+1}A_t
                    +\sum_i p_{t+1,i} f^i_{ux,t},\\
M_{uu,t}^{\mathrm{DDP}}&=c_{uu,t}+B_t^\top P_{t+1}B_t
                    +\sum_i p_{t+1,i} f^i_{uu,t}.
\end{aligned}
$$

Everything downstream is the same recipe with the new $M_t$: the formulas for
$k_t$, $K_t$, $p_t$ and $P_t$, the forward pass, the line search and the
regularization. In {prf:ref}`alg-trajectory-ilqr`, only step 3 changes.

DDP does not build a new local problem with quadratic dynamics and solve it
with Riccati. With quadratic dynamics the local problem would no longer be
LQR. DDP instead expands each backup to second order directly, and the extra
term lands in $M_t$.

````{prf:algorithm} Backward pass for DDP
:label: alg-ddp-backward-pass

**Input:** Jacobians $A_t,B_t$, transition Hessians $f^i_{zz,t}$ and cost derivatives $c_{z,t},c_{zz,t}$ along the current plan; terminal gradient $p_T$ and Hessian $P_T$.

**Output:** Feedforward terms $k_t$ and gains $K_t$ for $t=1,\ldots,T-1$.

1. For $t=T-1,\ldots,1$:
   1. Form $q_t=c_{z,t}+F_t^\top p_{t+1}$ and $M_t=c_{zz,t}+F_t^\top P_{t+1}F_t+\sum_{i=1}^{n_x}p_{t+1,i}\,f^i_{zz,t}$.
   2. Solve $k_t=-M_{uu,t}^{-1}q_{u,t}$ and $K_t=-M_{uu,t}^{-1}M_{ux,t}$.
   3. Update $p_t=q_{x,t}-M_{ux,t}^\top M_{uu,t}^{-1}q_{u,t}$ and $P_t=M_{xx,t}-M_{ux,t}^\top M_{uu,t}^{-1}M_{ux,t}$.
2. Return $k_t$ and $K_t$.
````

The only line that differs from {prf:ref}`alg-ilqr-backward-pass` is the
formula for $M_t$. The curvature choice can still change both the direction
of a step and the amount of regularization needed to accept it.

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

The boat's controls must remain in $[-1,1]^2$, even when the unconstrained
correction asks for more thrust. At zero state deviation, the feedforward step
is the solution of the box-constrained quadratic problem

$$
\min_k\ q_u^\top k+\frac12k^\top(M_{uu}+\mu I)k,
\qquad -\mathbf1-\bar{\mathbf u}_t\leq k\leq\mathbf1-\bar{\mathbf u}_t.
$$

For two controls, each coordinate is either free, fixed at its lower bound,
or fixed at its upper bound. The teaching implementation checks the nine
combinations, minimizes over each face, and keeps the best feasible candidate.
Positive-definite curvature makes this a strictly convex QP. Larger systems
can use an active-set solver rather than enumerate all combinations
{cite:p}`tassa2014control`.

The feedback rows of saturated controls are zero while their active set stays
fixed. For the free coordinates $\mathcal F$, solve
$(M_{uu}+\mu I)_{\mathcal F\mathcal F}K_{\mathcal F}
=-(M_{ux})_{\mathcal F}$.
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

(sec-thermoacoustic-pulldown)=
## Pulling Down a Thermoacoustic Refrigerator

A standing-wave thermoacoustic refrigerator is a sealed, gas-filled tube with
a loudspeaker at one end and a stack of thin parallel plates inside
({numref}`fig-thermoacoustic-geometry`a). The loudspeaker excites a standing
sound wave in the tube. Each gas parcel near a plate is pushed back and forth
along it, compressing and heating as it moves toward the loudspeaker end and
expanding and cooling as it moves away. At the loudspeaker end of its travel
the parcel is warmer than the plate and gives up heat to it; at the far end it
is cooler than the plate and takes heat from it. Every cycle therefore moves a
little heat one step along the plate, and the parcels together act as a
bucket brigade that transports heat from the far end of the stack toward the
loudspeaker end. Sound is used to move heat from a cold place to a warmer
one, against the direction it would flow on its own. A household heat pump
does the same thing with a compressor when it extracts heat from cold outdoor
air and delivers it indoors; a refrigerator is the same machine used for its
cold side. A cold heat exchanger at the far end of the stack connects to the
payload, the refrigeration term for the object or compartment being cooled,
and a hot heat exchanger at the loudspeaker end rejects the heat to ambient
air. There are no moving parts other than the loudspeaker, and the only knob
is how hard it drives. The control problem is what refrigeration engineers
call a pull-down: bringing the payload from ambient temperature down to its
set point after switch-on. Here that means choosing a driver-amplitude
schedule that brings the payload to a target temperature by a fixed deadline
while spending little acoustic energy.

:::{iframe} ../interactive/thermoacoustic-refrigerator.html
:label: fig-thermoacoustic-geometry
:width: 100%
:title: Thermoacoustic refrigerator: device, lumped model, and recorded pull-down plans
:class: thermoacoustic-replay
:placeholder: _static/thermoacoustic_pulldown/geometry.svg

Geometry of the thermoacoustic refrigerator and its two-state abstraction.
(a) The device: a loudspeaker piston at the left end of a sealed resonator, a
hot heat exchanger, a stack of thin plates, and a cold heat exchanger. The
dashed envelope is the standing-wave pressure amplitude. Gas parcels
oscillating along the plates pump heat from the cold exchanger toward the
hot one. (b) The lumped model the optimizer sees. The state is the pair of
exchanger temperatures, the action is the driver amplitude, the stage cost
charges acoustic work, and the terminal cost penalizes the final cold-side
temperature error. In the online book the figure replays the recorded
iLQR and DDP plans: the piston stroke, wave, and parcel motion follow the
action, the exchanger and node tints follow the state, and the panels
below compare the plan with a constant-amplitude schedule. The static
version shows the same two panels.
:::

In the online book, the replay reads the recorded plans and integrates only
the constant-amplitude comparison; it does not rerun the optimizer.

Rather than simulate the acoustics, we lump the whole device into two
thermal masses ({numref}`fig-thermoacoustic-geometry`b): the cold exchanger
together with its payload, and the hot exchanger. The state is their
temperatures, $\mathbf x_t=(T_{c,t},T_{h,t})$, and the action is the
normalized driver amplitude $u_t\in[0,1]$, so this is a two-dimensional
continuous-state, one-dimensional continuous-action problem. Let
$\Delta T=T_h-T_c$ be the temperature span across the stack and
$\eta(\Delta T)=1-\Delta T/\Delta T_0$ a pumping efficiency that decreases
linearly with that span. The continuous-time dynamics are

$$
\begin{aligned}
\dot Q_c &= k_q u^2\eta(\Delta T),&
\dot W &= k_w u^2\eta(\Delta T)+k_vu^2+k_3u^3,\\
C_c\dot T_c &= -\dot Q_c+Q_{\mathrm{load}},&
C_h\dot T_h &= \dot Q_c+\dot W-UA(T_h-T_{\mathrm{amb}}).
\end{aligned}
$$

Here $\dot Q_c$ is the heat pumped out of the cold side, $\dot W$ is the
acoustic work the driver injects, $Q_{\mathrm{load}}$ is a parasitic heat
leak into the payload, and $UA$ is the hot exchanger's conductance to
ambient. Both heat flows scale with $u^2$ because acoustic power is
quadratic in wave amplitude, and both weaken as the span grows; this is the
short-stack approximation of thermoacoustics
{cite:p}`swift1988thermoacoustic,swift2017thermoacoustics`. Two loss terms
complete the work: a viscous term in $u^2$ and a cubic term $k_3u^3$ that
makes hard driving disproportionately expensive. The coupling matters for
the optimizer. Driving hard pumps heat faster, but it also dumps that heat
plus the work into the hot side, which raises $T_h$, widens the span, and
lowers the efficiency of every later step. Refrigeration engineers measure
this efficiency by the coefficient of performance (COP), the ratio
$\dot Q_c/\dot W$ of heat pumped to work spent while the driver is on.
Unlike an efficiency it can exceed one, because the work only moves heat
rather than producing cold. In this model the COP falls as the span grows
and, through the cubic term, as the amplitude grows; one scenario below
removes the cubic term so that the COP no longer depends on amplitude. All
coefficients are synthetic teaching values, not measurements of a particular
device.

| Quantity | Value |
| :--- | ---: |
| Cold and hot heat capacities $C_c,C_h$ | $20,50\ \mathrm{J/K}$ |
| Hot conductance $UA$ | $0.5\ \mathrm{W/K}$ |
| Ambient temperature $T_{\mathrm{amb}}$ | $20\ ^\circ\mathrm C$ |
| Parasitic cold-side load $Q_{\mathrm{load}}$ | $1\ \mathrm W$ |
| Pumping coefficient $k_q$ | $5\ \mathrm W$ |
| Work and viscous coefficients $k_w,k_v$ | $4,1\ \mathrm W$ |
| Cubic-loss coefficient $k_3$ | $2\ \mathrm W$ |
| Critical temperature span $\Delta T_0$ | $40\ \mathrm K$ |
| Target temperature $T^\star$ | $5\ ^\circ\mathrm C$ |

Both temperatures begin at ambient, $20\ ^\circ\mathrm C$. A fourth-order
Runge--Kutta step of length $h=1\ \mathrm s$ defines the deterministic
transition $\mathbf x_{t+1}=\mathbf f_t(\mathbf x_t,u_t)$ over a horizon
of $T-1=300$ actions, so the episode is exactly five minutes. The cost uses
the same stage-plus-terminal form as the docking problem:

$$
c_t(\mathbf x_t,u_t)=h\,w_E\dot W(\mathbf x_t,u_t),\qquad
c_T(\mathbf x_T)=w_T(T_{c,T}-T^\star)^2,
\qquad w_E=0.05,\quad w_T=10.
$$

The stage cost charges only energy, and the target enters only through the
terminal cost, so the objective is sparse in the RL sense: nothing tells the
optimizer how cold to be at intermediate times. The terminal penalty is
soft, so a plan can trade some arrival error for lower energy. Both methods
start from the constant open-loop sequence $u_t=0.5$ and use the same action
bounds and stopping settings.

:::{include} artifacts/thermoacoustic_pulldown/results.md
:::

:::{figure} _static/thermoacoustic_pulldown/pulldown.svg
:label: fig-thermoacoustic-pulldown
:width: 100%

Cold and hot temperatures under the baseline plan, with a constant
full-amplitude rollout for comparison. The lower panel shows their driver
amplitudes. The target line and temperature traces show the tradeoff between
energy use and final cold temperature.
:::

The baseline plan drives relatively hard at first, eases off, then ramps to
full amplitude near the deadline. Early driving uses the initially small
temperature span, but it also warms the hot side and weakens later pumping.
The fixed load adds the same total heat over every 300-second plan. Control
timing still changes the temperature span and therefore the final cold
temperature. Running at full amplitude throughout uses about
$1426\ \mathrm J$ and reaches
$2.20\ ^\circ\mathrm C$; its nonlinear cost is $149.886$, versus $51.224$ for
the baseline local plan. The soft terminal cost need not favor the coldest
possible final state.

:::{figure} _static/thermoacoustic_pulldown/variants.svg
:label: fig-thermoacoustic-variants
:width: 100%

Final computed amplitude schedules for the baseline, no-load,
sluggish-hot-side, and no-cubic-loss cases. The first three are iLQR plans;
the no-cubic-loss plan is from DDP. Removing the cold-side load flattens the
schedule, while changing the hot-side dynamics produces a long ramp.
:::

Without the parasitic load, a nearly constant amplitude is economical. To
see the tendency, temporarily hold the temperature span fixed and write
$v=u^2$. Heat pumped is then proportional to $v$, while the cubic loss is
proportional to $v^{3/2}$, a convex function for $v\geq0$. For a fixed total
amount of pumping, Jensen's inequality favors spreading $v$ through time.
The actual schedule is not exactly constant because the span and hot-side
temperature still respond to the controls. With $C_h=200\ \mathrm{J/K}$ and
$UA=0.2\ \mathrm{W/K}$, the amplitude rises through most of the horizon, then
turns off for the final two intervals. Both thermal storage and heat rejection
changed in that experiment, so it does not isolate the effect of either
parameter.

The [DDP curvature term](#sec-ddp-curvature) also has a direct physical
interpretation here. For the continuous cold-side equation,

$$
\frac{\partial^2\dot T_c}{\partial u\,\partial\Delta T}
=\frac{2k_qu}{C_c\Delta T_0}.
$$

Increasing the temperature span weakens pumping, and this mixed derivative
measures how that effect changes with amplitude. The solver differentiates
the complete RK4 transition, whose Hessian also contains effects from the
intermediate integration stages.

:::{figure} _static/thermoacoustic_pulldown/convergence.svg
:label: fig-thermoacoustic-convergence
:width: 100%

Actual nonlinear cost after each accepted update for iLQR and DDP in the
baseline and no-cubic-loss cases. The vertical axis is logarithmic. The
curves compare these local solves from the same initial control sequence,
not a general ranking of the methods.
:::

When $k_3=0$, both pumping and work are quadratic in amplitude at a fixed
temperature span. Their instantaneous ratio then no longer depends on
amplitude, leaving weaker preferences among some schedules. In the recorded
run, iLQR reaches its iteration limit while DDP satisfies the stopping test;
both reach nearly the same final cost. For the sluggish hot side, DDP reaches
a different, slightly lower-cost local plan than iLQR. Lowering the target to
$0\ ^\circ\mathrm C$ produces a plan saturated at $u_t=1$ throughout the
fixed horizon, exercising the box-constrained backward step. These outcomes
depend on the initial sequence. Starting the baseline from constant
amplitudes $0.1$ and $0.9$ leads to other local plans, so neither result
certifies a global minimum.

(sec-radiotherapy-fractionation)=
## Scheduling Radiotherapy Fractions

Radiotherapy rarely delivers its dose in one sitting. A course is split into
fractions, typically one per weekday over several weeks, because healthy
tissue repairs part of its radiation damage between sessions while many
tumors repair less. Every beam that reaches the tumor also crosses healthy
organs on its way in and out, so a treatment plan must decide both how hard
to irradiate on each day and from which directions. With one control vector
per session and a state that summarizes the damage accumulated so far, this
is again a finite-horizon optimal-control problem, and iLQR applies to it
directly.

The example below is a synthetic slice of the chest with a tumor at the
center and three organs around it: the spinal cord, the heart, and a lung.
Three beams aim at the tumor, and each one crosses exactly one organ before
reaching it ({numref}`fig-radiotherapy-replay`). The control $\mathbf u_t\in\mathbb
R^3$ holds the three beam intensities of session $t$, each between $0$ and
$3\ \mathrm{Gy}$. The beams are scaled so that intensity $u_{t,j}$ adds
$u_{t,j}$ gray to the tumor's mean dose. The mean dose received by structure $k$
is $d_{t,k}=\mathbf a_k^\top\mathbf u_t$, where the dose matrix with rows
$\mathbf a_k$ is computed once from the beam geometry. The tumor row is
$(1,1,1)$, and each organ row has a single large entry, between $1.28$ and $1.50$, for the beam
that crosses it. That beam enters through the organ before it has lost
intensity in the tissue, so the organ receives more than the tumor does.

:::{iframe} ../interactive/radiotherapy-fractionation.html
:label: fig-radiotherapy-replay
:width: 100%
:title: Radiotherapy fractionation: recorded iLQR and DDP plans
:class: radiotherapy-replay
:placeholder: _static/radiotherapy_fractionation/plan.svg

The fractionation problem and its optimized plan. Left: the synthetic slice,
with the course dose of the final DDP plan. Arrows mark where the three beams
enter. Top right: the dose that each session delivers to the tumor, split by
beam, against the uniform 3 Gy plan (dashed). Bottom right: health of the
tumor and the three organs under the optimized plan (solid) and the uniform
plan (dotted). In the online book the figure replays the recorded plans
session by session and lets the reader scrub through the accepted iLQR and
DDP iterates.
:::

The state is a health score for each structure, following the convention of
the clinical planning work this example is adapted from
{cite:p}`fu2019convex`. The tumor starts at $h_{1,0}=1$ and is controlled
when its health reaches $0$. Each organ starts at $0$, and its health becomes
negative as damage accumulates; $-1$ is its tolerance. A session damages
structure $k$ according to the linear-quadratic model of radiobiology
{cite:p}`fowler1989linear`, and the days until the next session then let the
organs repair and the tumor regrow:

$$
h_{t+1,k}=\Big(h_{t,k}-\frac{\alpha_kd_{t,k}+\beta_kd_{t,k}^2}{E_k}\Big)\rho_k(t),
\qquad
\rho_k(t)=\begin{cases}e^{\lambda g_t},&k=\text{tumor},\\ e^{-r_kg_t},&k\ \text{an organ}.\end{cases}
$$

Here $g_t$ is the number of days until the next session: one from Monday to
Thursday and three on Friday. The quadratic term $\beta_kd^2$ makes one large
fraction more damaging than two half-size fractions with the same total dose.
Late-responding healthy tissue has a larger $\beta$ relative to $\alpha$ than
most tumors, and that difference makes splitting the course worthwhile. The
coefficients below are synthetic teaching values with plausible ratios, not
clinical parameters.

| Structure | $\alpha_k$ (1/Gy) | $\beta_k$ (1/Gy²) | Effect scale $E_k$ | Daily repair $r_k$ or regrowth $\lambda$ |
| :--- | ---: | ---: | ---: | ---: |
| Tumor | 0.30 | 0.030 | 22 | regrowth 0.005 |
| Spinal cord | 0.10 | 0.050 | 2 | 0.01 |
| Heart | 0.10 | 0.030 | 2 | 0.04 |
| Lung | 0.10 | 0.020 | 2 | 0.10 |

The cost charges a small amount for beam intensity, penalizes organ health
below tolerance at every session, requires the tumor to reach zero at the end
of the course, and rewards organ health left at the end:

$$
\begin{aligned}
c_t(\mathbf h,\mathbf u)&=\tfrac{0.01}{2}\|\mathbf u\|^2+50\sum_{k\ne\text{tumor}}s(-h_k-1),\\
c_T(\mathbf h)&=200\,s(h_{\text{tumor}})^2+50\sum_{k\ne\text{tumor}}s(-h_k-1)-\sum_{k\ne\text{tumor}}h_k,
\end{aligned}
$$

where $s(z)=\log(1+e^{25z})/25$ is a smooth version of $\max(z,0)$, as in the
docking costs. The course has $T-1=20$ sessions over four weeks. Both
solvers start from equal beams of $0.7\ \mathrm{Gy}$, a uniform plan of
$2.1\ \mathrm{Gy}$ per session that leaves the tumor uncontrolled, and use
the same box-constrained backward pass as the docking problem.

:::{include} artifacts/radiotherapy_fractionation/results.md
:::

The simplest curative alternative gives every session equal beams, raised
until the tumor just reaches zero. That takes $3\ \mathrm{Gy}$ per session and
$60\ \mathrm{Gy}$ in total, and it drives the spinal cord to a health of
$-2.26$, more than twice its tolerance; the heart also ends just past its
limit. The optimized plan controls the tumor with $56\ \mathrm{Gy}$ and keeps
every organ within tolerance. Three features of the schedule in
{numref}`fig-radiotherapy-replay` follow from the model. The beam through the
lung carries more than half of the dose, and the beam through the cord the
least, because the lung repairs fastest and has the smallest $\beta$ while
the cord repairs slowest and has the largest. The doses are front-loaded:
early damage to an organ has more days to be repaired before the end of the
course, and early tumor kill leaves less tumor to regrow. Fridays receive
extra dose because three days of repair follow them. The organ health curves
level off after the first week, where each session's new damage is roughly
balanced by repair. Clinical courses are not front-loaded this way; the
pattern reflects the first-order repair model and the absence of acute
toxicity limits, which a clinical planning model would include.

:::{figure} _static/radiotherapy_fractionation/iterations.svg
:label: fig-radiotherapy-iterations
:width: 100%

Left: actual nonlinear cost after each accepted update for iLQR and DDP from
the same uniform 2.1 Gy plan, on a logarithmic axis. Right: the dose that
selected DDP iterates deliver to the tumor in each session. The first update
pushes dose to the last sessions, where it is most directly tied to the
terminal tumor cost; later updates move it to the start of the course and to
the Fridays.
:::

Both methods reach the same plan, but DDP needs 11 accepted updates and iLQR
53. The transition explains the difference. It is linear in the health
state and quadratic in the beams, so the only second-order transition term
that DDP adds to the [backward recursion](#sec-ddp-curvature) is

$$
\sum_kp_{t+1,k}\,\frac{\partial^2h_{t+1,k}}{\partial\mathbf u_t^2}
=-2\sum_kp_{t+1,k}\,\frac{\beta_k\rho_k(t)}{E_k}\,\mathbf a_k\mathbf a_k^\top,
$$

where $p_{t+1,k}$ is the derivative of the cost-to-go with respect to
$h_{t+1,k}$. Along the final plan this derivative is positive for the tumor,
whose remaining health costs, and negative for each organ, whose health is
rewarded. The organ terms therefore add positive curvature: they tell the
backward pass that pushing a beam harder damages its organ more than in
proportion. The tumor term subtracts curvature, because a larger fraction
also kills the tumor more than in proportion. iLQR drops both terms and
treats the effect of each extra gray as fixed around the current plan, so it
needs more updates to discover the advantage of spreading dose over the
sessions and beams.

## Exercises

:::{exercise} Backward induction by hand
:label: ex-ilqr-two-step

Consider a scalar problem with $T=3$, initial state $x_1=0$, dynamics
$x_{t+1}=x_t+u_t$, and cost

$$
J=\frac12u_1^2+\frac12u_2^2+\frac\beta2(x_3-r)^2,\qquad \beta>0.
$$

1. Treat $x_2$ as given, substitute $x_3=x_2+u_2$, and minimize over $u_2$.
   Write the minimizing $u_2$ as a function of $x_2$ and compute the
   remaining cost-to-go.
2. Use $x_2=u_1$ to minimize over $u_1$. Find both controls and the terminal
   state.
3. For $\beta=r=1$, check that $u_1=u_2=1/3$, $x_3=2/3$, and $J=1/6$. Why is
   the final position short of $r$?
4. Why is it useful that step 1 returns $u_2$ as a function of $x_2$ rather
   than as a number? What happens as $\beta\to\infty$?
:::

:::{solution} ex-ilqr-two-step
:class: dropdown

The last control is $u_2=\beta(r-x_2)/(1+\beta)$, and its minimized tail
is $\beta(x_2-r)^2/[2(1+\beta)]$. With $x_1=0$, minimizing the remaining
expression gives $u_1=\beta r/(1+2\beta)$, followed by the same value for
$u_2$. Thus $x_3=2\beta r/(1+2\beta)$. For $\beta=r=1$ this gives
$u_1=u_2=1/3$ and $x_3=2/3$, so $J=\tfrac1{18}+\tfrac1{18}+\tfrac1{18}=1/6$.
The final position falls short because the objective trades terminal error
against effort; arrival is penalized, not imposed as an equality.

The function $u_2(x_2)$ is a feedback law: if the state reaching the last step
changes, the minimizing last control changes with it. In the vector case this
dependence becomes the gain $K_t$. As the terminal weight grows, the two
controls tend to $r/2$ and the terminal state tends to $r$.
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

For a positive-definite $2\times2$ matrix $M$ with nonzero off-diagonal entries,
compare the exact box-QP solution with clipping $-M^{-1}q$ componentwise.
Explain why they can differ. What should the trajectory optimizer retain if
every nonlinear trial is rejected?
:::

:::{solution} ex-ilqr-saturation
:class: dropdown

Fixing one coordinate at a bound changes the stationarity equation for the
other coordinate through the off-diagonal entry of $M$. Componentwise
clipping does not solve that new equation. The face minimization does.
If no nonlinear trial is accepted, the solver retains its previous controls
and states and reports that it could not make progress; it must not replace
them with an unaccepted trial or report convergence from rejection alone.
:::

:::{exercise} Thermal load and amplitude scheduling
:label: ex-ilqr-pulldown-load

Set the parasitic cold-side load to zero and rerun the baseline refrigerator
problem. Explain why the optimized amplitude becomes nearly constant, using
the convexity of the cubic loss. Then restore the load and double the horizon
to $600\ \mathrm s$. Predict how the amplitude schedule and energy use change.
:::

:::{solution} ex-ilqr-pulldown-load
:class: dropdown

With no load, cooling obtained early is not directly undone by a constant
heat leak. If the temperature span were fixed, a given amount of cooling
would fix the sum of $v_t=u_t^2$. The cubic loss is proportional to
$v_t^{3/2}$, so Jensen's inequality favors equal $v_t$ across time. The
temperature span still evolves, which explains the modest variation in the
computed schedule. With the load restored, the longer horizon admits lower
amplitude over more intervals, but the leak adds heat throughout that extra
time. In this model, the $600\ \mathrm s$ local solution starts near $0.59$,
reaches $0.89$ by $550\ \mathrm s$, and uses full amplitude in the final
intervals. Its approximately $1392\ \mathrm J$ of energy exceeds the shorter
baseline's use.
:::

## Computational Sources

The implementation separates the generic backward recursion from the physical
models and artifact builders:

- {download}`Shared iLQR and DDP solver <code/trajectory_optimization.py>`
- {download}`Boat dynamics, costs, and validation <code/boat_docking.py>`
- {download}`Experiment and figure builder <scripts/build_boat_docking_artifacts.py>`
- {download}`Tube figure and animation builder <scripts/build_boat_tube_figure.py>`
- {download}`Numerical diagnostics and source hashes <artifacts/boat_docking/metrics.json>`
- {download}`Thermoacoustic model and costs <code/thermoacoustic_pulldown.py>`
- {download}`Thermoacoustic geometry figure <code/thermoacoustic_geometry.py>`
- {download}`Thermoacoustic experiment builder <scripts/build_thermoacoustic_pulldown_artifacts.py>`
- {download}`Thermoacoustic diagnostics <artifacts/thermoacoustic_pulldown/metrics.json>`
- {download}`Thermoacoustic replay data <interactive/thermoacoustic-refrigerator-data.json>`

Run `uv run python scripts/build_boat_docking_artifacts.py` from the repository
root to reproduce the solves, figures, results table, and browser data. Normal
book builds read those artifacts. Tests independently check the derivatives,
the composed DDP curvature, the scalar calculation, and agreement between
backward elimination and a dense quadratic solve.

Run `uv run python scripts/build_thermoacoustic_pulldown_artifacts.py` to
regenerate the refrigerator results table, figures, diagnostics, and the
browser replay data.

## Summary and Outlook

A feasible nonlinear rollout supplies the point around which dynamics and
costs are approximated. Backward elimination of a local quadratic problem
then supplies both a feedforward change and a state-dependent correction.
iLQR repeats that calculation with linearized dynamics; DDP includes the
second-order chain-rule terms from the transition. Nonlinear rollouts,
regularization, and acceptance tests connect these local calculations to a
decreasing sequence of actual trajectory costs.
The refrigerator applies the same calculation to a bounded driver amplitude,
with energy use and terminal temperature competing in the objective.

The quadratic tail produced by elimination is a local model of the cost-to-go
along one trajectory. For linear-quadratic problems it is the
[optimal cost-to-go](#sec-lqr-dp) itself. When the model prescribes a
distribution over successors instead of one next state,
[stochastic dynamic programming](stochastic-dp.md) replaces the continuation
cost by a conditional expectation.
