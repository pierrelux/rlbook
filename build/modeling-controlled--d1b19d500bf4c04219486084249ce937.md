---
description: Formulate sequential decision problems by identifying their state, action channel, uncertainty, observations, constraints, and available model interface.
kernelspec:
  name: python3
  display_name: Python 3
---
# Controlled Systems: Boundaries, States, and Actions

The introduction identified the state, action, evolution, objective, and
constraints as the ingredients of a sequential decision problem. Before any
algorithm can compare actions, how can those ingredients be tied to one
physical system and one decision maker?

A simulated playground swing can be made easy to control by placing a motor at
the suspension pivot. A rider on a real swing has no such motor. The rider can
change body shape, and the chain can pull but cannot push. An optimizer may
produce a successful trajectory for the first model even though neither its
action nor its suspension exists in the target system.

This failure occurs before the choice of control or learning algorithm. A
**decision model** must say what evolves, what can be observed, which
interventions are possible, how uncertainty enters, and which constraints must
hold. Its **information pattern** specifies what the decision maker knows when
each action is selected. These choices lead from physical systems to
state-space equations and then to simulators, logged transitions, and
interactive environments. Each case study tests one part of the resulting
model.

Trajectory optimization, model predictive control, dynamic programming, and
reinforcement learning all reason about the consequences of actions. They can
only compare consequences that the model makes possible, including the predicted
outcome of an action that has not yet been tried. Such an unobserved alternative
is called a **counterfactual**. A fictitious actuator creates infeasible plans,
an incomplete state destroys predictive information, and a logged dataset
cannot answer counterfactual questions about actions it does not cover.
Modeling therefore determines which later computations and claims are
meaningful.

The modeling choices have a useful dependency order. The system boundary first
separates internal variables from external influences. That boundary determines
the state, observation, action, and disturbance. Dynamics then describe how
these quantities interact, while the objective and constraints define the
decision problem. Finally, the available model interface determines whether an
algorithm can differentiate equations, run new simulations, or only analyze a
fixed log. The chapter follows this order.

::::{admonition} Learning goals
:class: note

After reading this chapter, you should be able to:

- formulate a sequential decision problem by specifying its boundary, state,
  action, disturbance, observation, objective, and constraints;
- locate the physical channel through which an action changes the system;
- write deterministic and stochastic state-space models in continuous and
  discrete time;
- distinguish state from observation and an open-loop action sequence from a
  feedback policy;
- determine which operations are supplied by equations, simulators, logged
  transitions, and interactive environments;
- separate known structure from learned components and design a controlled
  experiment that can expose a modeling error.
::::

::::{admonition} Prerequisites
:class: tip

The chapter assumes linear algebra, multivariable calculus, elementary
probability, and the meaning of an ordinary differential equation. Numerical
integration is reviewed in [Solving Initial Value Problems](appendix_ivps.md).
::::

## System Boundaries and Action Channels

Which physical variables belong inside the controlled system, and which
interventions can the decision maker actually apply across that boundary?

A playground swing, an overhead crane, and a wave-energy converter all
oscillate. Their visual similarity does not make them the same control problem.
A rider changes body shape, a crane accelerates the suspension point, and a
wave-energy converter changes a dissipative load while the sea supplies the
forcing.

The boundary identifies which physical components belong to the system being
modeled and which influences arrive from outside it. The action channel then
identifies the intervention available to the decision maker within that
boundary. These two choices must precede the equations because they determine
which input terms the equations are allowed to contain.

:::{exercise} Locate the action channel
:label: ex-dynamics-opening-action-channel

Before reading the table, predict which system should create oscillation, which
should suppress it, and which should retain motion to extract energy. Then name
the physical quantity that each controller can change.
:::

```{code-cell} python
:tags: [remove-cell]

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from modeling_interfaces import make_overview_figure
```

```{code-cell} python
:tags: [remove-input]
:label: fig-modeling-action-channels
:caption: The systems share an oscillatory silhouette but expose different control interfaces. The rider changes internal shape, the crane accelerates its pivot, and the wave device changes a nonnegative dissipative load. Their desired motions and feasible actions therefore differ.

overview_figure = make_overview_figure()
display(overview_figure)
plt.close(overview_figure)
```

| system | state must describe | action changes | external influence | objective | hard constraint |
|---|---|---|---|---|---|
| playground swing | suspension angle and rider pose | internal body shape | gravity and drag | add mechanical energy | a chain can pull but cannot push |
| overhead crane | trolley position and load angle | trolley acceleration | gravity and damping | move the load and remove sway | acceleration is bounded |
| wave-energy converter | flap angle and velocity | power-take-off damping | incident waves | extract energy within motion limits | passive damping is nonnegative |

:::{solution} ex-dynamics-opening-action-channel
:class: dropdown

The rider creates oscillation by changing body shape. The crane suppresses load
sway by changing trolley acceleration. The wave-energy converter retains motion
while extracting energy by changing nonnegative power-take-off damping. The
systems look similar, but none exposes an arbitrary force at the oscillating
coordinate.
:::

The three systems need different inputs, but their mechanical equations can be
organized by the same bookkeeping template. Let $q$ collect the independent
coordinates needed to describe configuration, such as an angle or a trolley
position. Such coordinates are called **generalized positions**. Their
velocities are $v=\dot q$; $z$ collects additional actuator or environment
states; and $u$ is the commanded input. A broad finite-dimensional model is

$$
\dot q=v,
\qquad
M(q,z)\dot v+c(q,v,z,\dot z)+\nabla_q V(q,z)+r(q,v,z)
=B(q,v,z)u+G(q,v,z)\xi_v(t),
\qquad
\dot z=g(q,v,z,u,\xi_z(t)).
$$

The mass matrix $M$ maps coordinate accelerations to their corresponding forces
or torques, collectively called **generalized forces**. It is positive definite,
which ensures that every nonzero velocity has positive kinetic energy.
The term $c$ collects velocity- and configuration-dependent inertial forces,
including Coriolis and centrifugal effects. The gradient $\nabla_q V$ appears
on the left side, so the corresponding conservative force is
$-\nabla_q V$. The term $r$ represents dissipative resistance. The columns of
$B$ and $G$ specify the physical directions through which actions $u$ and
external inputs $\xi=(\xi_v,\xi_z)$ enter. Dissipation requires
$v^\top r(q,v,z)\geq0$, so resistance cannot add mechanical energy.
An action can enter directly through $Bu$ or indirectly by changing $z$. A
moving coordinate whose inertia is modeled belongs in $q$; a fixed parameter
or prescribed signal does not need its own state equation.

The term $B(q,v,z)u$ cannot be inferred from the word "action" alone. An
unconstrained vector input would omit the swing's internal-actuation geometry,
the crane's acceleration limit, and the wave device's passivity restriction.

:::{sidebar} Optional connection: port-Hamiltonian models
The mechanical template above is one coordinate-level instance of a broader
energy-based description. A port-Hamiltonian model writes

$$
\dot x=[J(x)-R(x)]\nabla H(x)+G(x)u,
\qquad y=G(x)^\top\nabla H(x),
$$

where $H$ stores energy, the skew-symmetric matrix $J$ routes energy, $R$
dissipates it, and $(u,y)$ form an external power port. The same structure can
connect mechanical, electrical, fluid, and thermal components. When $H$ has no
explicit time dependence, its energy balance is

$$
\dot H=-\nabla H(x)^\top R(x)\nabla H(x)+y^\top u.
$$

The first term dissipates stored energy, while $y^\top u$ is power supplied
through the port {cite}`vanDerSchaftJeltsema2014`. This chapter uses the more
concrete coordinate form.
:::

The system boundary also determines which variables are external. At sea, the
incident wave torque is an uncommanded disturbance to the energy converter. In
a laboratory wave tank, an experimenter can command a motorized paddle that
generates the waves. The paddle command is then an action, while the resulting
hydrodynamic torque remains a force on the flap. The converter has not changed;
the modeled boundary, decision maker, and physical actuator have.

Once the boundary and action channel are fixed, the remaining internal
variables must be summarized in a form that predicts what happens next. That
predictive summary is the state.

## State-Space Models

Once the boundary and action channel are fixed, which variables summarize the
past well enough to predict the effect of the next action?

A state summarizes the past information needed to predict future evolution
under a chosen action. The word "needed" depends on the model. For example,
indoor air temperature alone may fail to predict the next temperature if heat
stored in the walls is omitted. Two rooms with the same air temperature but
different wall temperatures respond differently after the heater is turned
off. Adding wall temperature to the state removes that ambiguity in a simple
thermal model.

Once the state has been chosen, it should make earlier history unnecessary for
one-step prediction. For a stochastic discrete-time model, this requirement is
the Markov property:

$$
\Pr(x_{t+1}\in A\mid x_{0:t},u_{0:t})
=\Pr(x_{t+1}\in A\mid x_t,u_t),
$$

for every set $A$ of possible next states. The left side conditions on the
entire state-action history, while the right side retains only the current
state and action. Their equality means that the older history supplies no
additional information about the next state. The definition is relative to the
declared model and time scale. A simulator may store a large internal state
while exposing a smaller observation to the controller, and that observation
need not itself be Markov.

A state-space model can keep disturbances and sensor errors as explicit inputs.
In discrete time, such a model has the form

$$
x_{t+1}=f_t(x_t,u_t,\xi_t),
\qquad
y_t=h_t(x_t,u_t,\nu_t).
$$

Here $f_t$ advances the state, while $h_t$ maps the state and action to the
sensor output $y_t$. The variable $\xi_t$ is a process disturbance that changes
the physical evolution, whereas $\nu_t$ is measurement noise that changes only
the reported observation. For fixed values of these inputs, both equations are
deterministic maps. The stochastic section later assigns probability laws to
them.

In continuous time, the corresponding equations are

$$
\dot x(t)=f(x(t),u(t),\xi(t)),
\qquad
y(t)=h(x(t),u(t),\nu(t)).
$$

The first equation specifies a rate of change rather than a one-step update.
Many physical laws are most compact in continuous time because mechanics,
circuit theory, fluid dynamics, heat transfer, and chemical kinetics describe
rates, flows, and conservation balances. Those fields provide reusable
structure: geometry, conservation laws, units, and admissible energy flows need
not be relearned from trajectories. Unknown parameters or empirical
relationships, such as drag or heat-transfer laws, can instead be estimated
inside that structure.

Continuous equations still need a sampling convention before a digital
controller can use them. Sensors and command interfaces operate at discrete
times even though the physical plant continues to evolve between updates.
Under a chosen hold rule, which specifies how an action is maintained between
updates, and a sampling period $\Delta t$, integrating the ODE defines a discrete transition
$x_{k+1}=F_{\Delta t}(x_k,u_k)$. The map $F_{\Delta t}$ is called the exact
flow over one sample interval. A numerical integrator approximates this map.
Changing $\Delta t$ or the hold changes the sampled transition, not the
underlying physical law. It can also change the disturbance model because a
short event may be visible on one sampling grid and absent from another.

:::{figure} _static/sampling-and-integration.svg
:label: fig-sampling-and-integration
:alt: A continuous scalar velocity model is sampled at fine and coarse periods. The fine zero-order-hold disturbance representation captures a short pulse, while the coarse left-endpoint representation misses it and predicts the wrong endpoint.

A sampling period and an interval integrator jointly define the discrete
transition. The action is held at $0.6\ \mathrm{m/s^2}$, and the true disturbance
acts only from $0.4$ to $0.6$ seconds. At $\Delta t=0.1$ seconds, its two sampled
intervals reproduce the $0.182\ \mathrm{m/s}$ endpoint contribution. At
$\Delta t=1$ second, the left-endpoint disturbance sample is zero, so the held
coarse model predicts $0.379\ \mathrm{m/s}$ instead of $0.562\ \mathrm{m/s}$.
The ODE and interval integrator are unchanged; the sampling period changes both
$F_{\Delta t}$ and what the sampled disturbance representation can express.
:::

Coarse output sampling alone does not erase a known event. An integrator given
the true event time and $\xi(t)$ can still resolve the pulse with internal
substeps. The miss in {numref}`fig-sampling-and-integration` occurs because the
coarse model replaces $\xi(t)$ by left-endpoint, zero-order-held samples.
Here zero-order hold means that each sampled value is kept constant until the
next sample.

The general nonlinear maps above can be difficult to analyze or optimize.
Linear state-space models provide a tractable local or exact special case:

$$
\dot x=Ax+Bu+E\xi,
\qquad
y=Cx+Du.
$$

The matrix $A$ describes autonomous state evolution, $B$ describes how the
action changes that evolution, and $E$ describes how disturbances enter. The
matrix $C$ selects or combines state components into measurements, while $D$
captures any immediate effect of the action on the measurement. Thus $B$ and
$C$ encode different questions. An actuator may be able to influence every
state direction without sensors measuring every state component. Conversely,
measuring the full state does not imply that the available actuators can move
it in every direction.

Linearity is a modeling choice rather than a claim that the world is linear at
all scales. A nonlinear system may be well approximated by a linear model near
an equilibrium. Dynamic programming itself is not restricted to linear models
or stabilization: its Bellman recursion also applies to nonlinear and
stochastic transitions. In the later inverted-pendulum example, a local linear
approximation supplies one stabilizing controller, while
[Trajectory Optimization](discrete-time-optimal-control.md) uses the nonlinear equations to move
the pendulum into that controller's local region.

Dynamics answer the counterfactual question "what happens after this action?"
A decision problem must additionally specify which consequences are preferred,
which are forbidden, and what information may be used when each action is
chosen.

## Decision Problems and Policy Classes

A state-space model predicts evolution under an input, but which objective,
constraints, and information pattern turn that prediction into a decision
problem?

In machine learning, a model often means a parameterized predictor. A
**decision model** contains more: a state, available actions, dynamics,
observations, an objective, constraints, and an information pattern. Prediction
remains one component, but decisions require counterfactual trajectories under
actions that may not yet have been observed.

For a finite horizon of $T$ actions, let $P_t(\cdot\mid x,u)$ be the
distribution of the next state after applying action $u$ in state $x$. The set
$\mathcal U_t(x)$ contains the feasible actions, and $\mathcal X_t$ contains
the admissible states. A stage cost $\ell_t(x,u)$ evaluates one decision, while
$\ell_T(x)$ evaluates the terminal state. A feedback policy $\pi_t$ maps the
available information $I_t$ to an action. These objects define the problem

$$
\begin{aligned}
\underset{\pi_0,\ldots,\pi_{T-1}}{\operatorname{minimize}}\quad
& \mathbb{E}^{\pi}\!\left[\sum_{t=0}^{T-1}
  \ell_t(x_t,u_t)+\ell_T(x_T)\right] \\
\text{subject to}\quad
& x_{t+1}\sim P_t(\cdot\mid x_t,u_t), \\
& u_t=\pi_t(I_t)\in\mathcal U_t(x_t), \\
& x_t\in\mathcal X_t \quad \text{almost surely}.
\end{aligned}
$$

The expectation averages the accumulated cost over trajectories generated by
the policy and stochastic dynamics. The state constraint holds **almost
surely**, meaning with probability one under that trajectory distribution.
Replacing it by a constraint on the average value, or by a **chance constraint**
that permits a stated probability of violation, would define a different
feasible set. Thus the formulation specifies what is optimized, how uncertainty
propagates, which information a policy may use, and which trajectories are
admissible. Omitting one of these pieces changes the problem even if the same
environment class and neural network are used afterward.

An **open-loop plan** selects the whole action sequence from information
available at the start,

$$
(u_0,\ldots,u_{T-1})=\mu(I_0).
$$

The sequence remains fixed after planning. A **feedback policy** instead
selects each action from the information available at that time,

$$
u_t=\pi_t(I_t).
$$

:::{figure} _static/open-loop-vs-feedback.svg
:label: fig-open-loop-vs-feedback
:alt: Two controllers face the same unexpected disturbance. The open-loop controller keeps its precomputed actions, while the feedback controller changes later actions after observing the displaced state.

An open-loop plan commits to future actions before the trajectory begins. A
feedback policy reevaluates its decision rule after each new observation. The
rule may be an optimizer, a learned policy, or a hand-written heuristic; the
word *feedback* describes what information reaches the action, not how the rule
was obtained. In this illustration,
$x_{k+1}=0.8x_k+u_k+\xi_k$: the open-loop schedule contains zeros, while the
feedback rule $u_k=-0.6x_k$ responds after the disturbance becomes visible at
$x_2$. The example shows information flow, not a guarantee that every feedback
rule performs well.
:::

A planned trajectory can perform well under its assumed initial condition and
disturbance forecast while reacting poorly to a tap, a delayed actuator, or an
unexpected arrival. Later chapters construct open-loop plans, turn them into
feedback by replanning, and compute state-contingent value functions.

The policy class and the model are separate choices. Several independent axes
describe the model itself.

| axis | common alternatives | consequence |
|---|---|---|
| time | continuous or discrete | differential equations or transition steps |
| evolution | deterministic or stochastic | one next state or a distribution over next states |
| state | continuous, discrete, or mixed | geometry of the state space and applicable solvers |
| action | continuous, discrete, or mixed | control authority and optimization method |
| observation | full state or partial/noisy measurement | state feedback or estimation from information histories |
| horizon | finite, infinite, or terminating | terminal conditions and objective definition |

Terms such as "continuous control" specify only one row of this table. They do
not determine the time representation, uncertainty, observation model, or
information available to the controller.

The swing example now combines these modeling choices. Its action is
continuous, its plant is simulated in continuous time, its feedback rule uses
observations at discrete times, and its suspension constraint changes the
equations when the chain becomes slack. The comparison isolates what goes
wrong when the action channel or constraint is modeled incorrectly.

(internal-actuation-in-swingrl)=
## Internal Actuation and Unilateral Constraints in SwingRL

How can a concrete model audit reveal that a nominally successful controller
relies on an actuator or contact force the real system cannot produce?

A rider who wants to swing higher cannot command a torque at the suspension
pivot. The available interventions are changes in body shape, and the chain can
transmit tension but not compression. A complete revolution provides a strict
test of both assumptions: the controller must inject enough energy through
internal motion while remaining feasible for a unilateral suspension.

The [`swing-rl`](https://github.com/pierrelux/swing-rl) environment turns those
requirements into a control problem. Its articulated standing model exposes two
normalized actions in $[-1,1]^2$: a squat target and a torso-lean target. The
action changes body geometry rather than applying a hidden torque at the
suspension pivot.

:::{figure} _static/swing-reduced-coordinates.svg
:label: fig-swing-reduced-coordinates
:alt: Geometry of the reduced swing model, showing the suspension angle theta, pivot-to-center-of-mass distance rho, center-of-mass offset alpha, absolute center-of-mass angle psi, and body angle beta.

The reduced coordinates describe the rider in aggregate rather than tracking
every joint. The suspension angle is $\theta$. The center of mass lies a
distance $\rho$ from the pivot and is offset by $\alpha$ from the suspension,
so its absolute angle is $\psi=\theta+\alpha$. The angle $\beta$ describes the
body's orientation relative to the suspension. Squatting mainly changes
$\rho$; leaning changes $\alpha$ and $\beta$.
:::

The reduced model makes the internal action channel visible in one equation.
Let $m$ be the rider's mass, $g$ gravitational acceleration, $\theta$ the
suspension angle, $\rho$ the center-of-mass distance from the pivot, $\alpha$
its offset from the suspension line, $\psi=\theta+\alpha$, $\beta$ the body
orientation relative to the suspension, and $J$ the rider's inertia about its
center of mass. With $\tau_{\mathrm{damp}}$ denoting the modeled damping torque,
the suspension-coordinate balance is

$$
(m\rho^2+J)\ddot\theta
=-m\rho^2\ddot\alpha
-2m\rho\dot\rho\dot\psi
-J\ddot\beta
-\dot J(\dot\theta+\dot\beta)
-mg\rho\sin\psi
+\tau_{\mathrm{damp}}.
$$

There is no commanded pivot torque on the right side. Torso lean instead
contributes through the acceleration terms in $\ddot\alpha$ and
$\ddot\beta$, while squatting changes $\rho$ and contributes through
$-2m\rho\dot\rho\dot\psi$. Multiplying this generalized force by the angular
velocity $\dot\theta$ gives its mechanical power. Near a bottom passage,
$\dot\psi\approx\dot\theta$, so the squat contribution is approximately

$$
\left(-2m\rho\dot\rho\dot\psi\right)\dot\theta
\approx -2m\rho\dot\rho\dot\theta^2.
$$

Shortening means $\dot\rho<0$. Since $m>0$, $\rho>0$, and
$\dot\theta^2\geq0$, the contribution is positive in either direction of
travel. The model therefore explains how a rider can add energy without a
fictitious pivot actuator.

The action channel is only half of the audit; the suspension must also obey the
correct force constraint. The matched comparison applies one
**phase-feedback law**, a rule that times squat and lean targets from the
observed oscillation phase, to the standing body under two suspension models
supplied by SwingRL. A
rigid rod can carry tension or compression. A chain can pull but cannot push.
The body parameters, initial state, squat and lean limits, feedback rule,
32-second horizon, and 0.02-second sampling interval remain fixed. During the
first 0.25 seconds, both runs multiply their requested squat and lean targets by

$$
r(t)=\min(t/0.25,1)
$$

so both commands rise linearly from zero to their full requested values instead
of jumping at the first step. This shared startup ramp prevents an initial
command discontinuity from being counted as a chain-release event.

Success requires the unwrapped suspension angle to change by at least $2\pi$,
one complete revolution. At that crossing, the pivot-to-seat distance must also
be at least $0.9L$, where $L$ is the fully extended suspension length. The
second test prevents a rotation from being credited while the seat is bunched
close to the pivot.

Both simulations use the same function from the current observation to squat
and lean commands. Once the chain releases, however, the rod and chain follow
different trajectories and produce different observations. Applying the same
feedback function to those different observations can produce different later
commands. The comparison therefore holds the feedback rule fixed, not the
entire open-loop command sequence.

The recorded comparison below applies that feedback rule to both suspension
models. The left plant permits axial force in either direction, while the right
plant enforces a chain that can only pull. Each plot displays the trajectory only
up to the selected playback time. The unwrapped angle keeps accumulating past
$\pm\pi$, so a full revolution remains visible. The compression-demand trace
shows how much pushing force the rod trajectory would require, a force the chain
cannot supply. The extension trace shows the pivot-to-seat distance relative to
$L$. Event buttons jump to times already located in the Python trajectory:
chain release, impact on reattachment (the snap), and the rod run's first
completed revolution.

```{code-cell} python
:tags: [remove-input]

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from swing_control import model_audit_player_html

display(HTML(model_audit_player_html(
    Path("_static/swing_modeling"),
    Path("artifacts/swing_modeling/events.json"),
    fallback_id="fig-swing-model-audit-fallback",
)))
```

:::{figure} _static/swing_modeling/model_audit.svg
:label: fig-swing-model-audit-fallback
:class: pdf-fallback
:alt: Matched SwingRL rod and chain keyframes with complete traces for angle, rod compression demand, and chain extension.

Matched keyframes and complete diagnostics from the same Python trajectories.
The online book adds the recorded 16:9 animation and event seeking.
:::

```{include} artifacts/swing_modeling/results.md
```

The rod completes a rotation, but part of that motion requires the suspension
to push the seat away from the pivot. A real chain cannot supply that push. It
goes slack when the pulling force would otherwise become negative, and the seat
and rider then move freely under gravity until the chain becomes fully extended
again. At that instant the chain snaps taut, the velocity changes abruptly,
and some mechanical energy is lost. In modeling language, this is a **hybrid
system**: it switches between taut and slack equations and includes a reset at
reattachment. Changing a coefficient in the always-taut rod equation cannot
create the missing free-flight mode or the snap.

This controller completes the task on the bidirectional-rod model and fails on
the unilateral-chain plant. The comparison does not rank control against
reinforcement learning, nor does it establish that a chain swing cannot be
rotated. It isolates a physical assumption: success on the rod model provides
no evidence that the same trajectory remains feasible when the suspension
cannot push.

A learned approximation fitted only to rod trajectories would inherit the same
blind spot. Data remain useful for estimating rider parameters, damping, and
contact losses after the unilateral mode structure is represented.
[Policy Gradients](policy-gradients.md) later evaluates PPO and the structured controller
against the same SwingRL action semantics and success criterion.

The swing audit changed a structural constraint while keeping the decision
rule fixed. The next modeling question concerns outcomes that vary even when
the state and action are fixed, such as demand arrivals or environmental
disturbances.

:::{dropdown} Inspect the shared SwingRL scenario and model audit
```{literalinclude} code/swing_control.py
:language: python
:start-at: class SwingScenario
:end-before: class SwingModelAuditAnimation
:linenos:
```

{download}`Download the complete SwingRL control example <code/swing_control.py>`

{download}`Download the recorded model audit <_static/swing_modeling/model_audit.mp4>`
:::

## Summary and Outlook

A controlled-system model fixes the system boundary, the variables that carry
state, the physical action channel, and the class of decision rules allowed to
use that state. The SwingRL audit shows why these choices precede optimization:
a controller cannot repair an action channel or contact mode that the model
omitted.

Deterministic state-space equations still assign one successor to each state
and action. What changes when disturbances make that successor random, or when
the controller observes only an ambiguous measurement? [Stochastic dynamics
and partial observation](stochastic-dynamics-observation.md) answer that
question.

## Exercises

:::{exercise} A complete decision model
:label: ex-dynamics-model-checklist

Choose a familiar sequential system and specify its boundary, state,
observation, action, disturbance, objective, information pattern, and one hard
constraint. Every item must refer to the same decision maker and time scale.
:::

:::{solution} ex-dynamics-model-checklist
:class: dropdown

Many answers are possible. For a room thermostat, one consistent model is:
the boundary contains the room, walls, and heater; the state contains indoor
air and wall temperatures; the observation is a noisy air-temperature reading;
the action is heater power; outdoor temperature and occupancy are disturbances;
the objective trades temperature error against energy; the information at time
$t$ is the observation and action history; and
$0\leq u_t\leq P_{\max}$ is a hard constraint. A different answer is valid when
all eight objects use one boundary and one sampling period.
:::

:::{exercise} Action channels are physical
:label: ex-dynamics-action-channels

In {numref}`fig-modeling-action-channels`, explain why replacing every input by
an unconstrained generalized force changes all three decision problems.
:::

:::{solution} ex-dynamics-action-channels
:class: dropdown

The replacement gives the swing a fictitious pivot actuator instead of
body-shape control and may bypass the chain's inability to push. It replaces
bounded trolley acceleration in the crane with direct forcing of the load. It
also lets the wave device inject signed force instead of choosing nonnegative
power-take-off damping. Each substitution changes the feasible trajectories,
rather than only the symbols used to describe them.
:::

:::{exercise} Sampling changes the transition
:label: ex-dynamics-euler-sampling

For $\dot x=-ax+bu$ with a zero-order-held input, derive the forward Euler
updates at step sizes $\Delta t$ and $\Delta t/2$. Which discrete coefficients
change with the sampling period?
:::

:::{solution} ex-dynamics-euler-sampling
:class: dropdown

At step size $\Delta t$,

$$
x_{k+1}=(1-a\Delta t)x_k+b\Delta t\,u_k.
$$

At step size $\Delta t/2$,

$$
x_{k+1}=\left(1-\frac{a\Delta t}{2}\right)x_k
{}+\frac{b\Delta t}{2}u_k.
$$

Both the state coefficient and the input coefficient change. Two half-steps
also compose a different finite-step approximation than one full Euler step,
although both converge to the same continuous flow as $\Delta t\to0$.
:::

:::{exercise} When shortening adds energy
:label: ex-dynamics-swing-power

Multiply $-2m\rho\dot\rho\dot\psi$ by $\dot\theta$ and use
$\dot\psi\approx\dot\theta$ near the bottom of the swing. Why can shortening
the rider's radius add energy in either direction of travel?
:::

:::{solution} ex-dynamics-swing-power
:class: dropdown

The approximate power contributed to the suspension coordinate is

$$
\left(-2m\rho\dot\rho\dot\psi\right)\dot\theta
\approx-2m\rho\dot\rho\dot\theta^2.
$$

Because $m,\rho>0$ and shortening gives $\dot\rho<0$, this term is positive.
The square removes the sign of $\dot\theta$, so the conclusion holds for
clockwise and counterclockwise passages.
:::

:::{exercise} A taut-slack hybrid model
:label: ex-dynamics-hybrid-swing

Describe the swing's taut and slack phases as a hybrid model. Identify the
continuous state, discrete mode, release guard, reattachment guard, and impact
reset; a complete equation of motion is not required.
:::

:::{solution} ex-dynamics-hybrid-swing
:class: dropdown

Use a continuous mechanical state containing positions, velocities, and rider
configuration, together with a mode
$m\in\{\text{taut},\text{slack}\}$. The taut mode enforces the chain-length
constraint and requires nonnegative tension. Release occurs when the tension
needed to maintain that constraint reaches zero and would become negative. The
slack mode follows unconstrained ballistic dynamics. Reattachment occurs when
the pivot-to-seat distance reaches the chain length with outward radial
velocity. The impact reset removes the inadmissible radial velocity and
dissipates its associated energy.
:::

:::{exercise} What rod-only data cannot identify
:label: ex-dynamics-rod-data

The structured controller completes a revolution on the rod model and fails on
the chain model. Why can a larger neural transition model fitted only to rod
trajectories fail to resolve this discrepancy?
:::

:::{solution} ex-dynamics-rod-data
:class: dropdown

Rod trajectories contain no release, slack flight, reattachment, or impact.
Additional function capacity cannot identify a mode absent from the data.
Chain interventions or an explicit unilateral hybrid structure must first
supply that missing behavior; data can then estimate parameters within it.
:::

:::{exercise} A one-factor SwingRL audit
:label: ex-dynamics-swing-intervention

Extend the SwingRL comparison with the seated controller, or disable either
squat or lean in the standing controller. Before running it, predict one
diagnostic that should change. Then explain which modeling assumption the
intervention tests.
:::

:::{solution} ex-dynamics-swing-intervention
:class: dropdown

A valid audit changes only one factor, records a prediction first, and compares
the result with the baseline. The seated controller changes rider morphology
and available actuation. Disabling squat removes the
$-2m\rho\dot\rho\dot\theta^2$ power channel; disabling lean removes the driven
angular terms. Peak unwrapped angle, revolution success, action-channel work,
and the timing of release or reattachment are suitable diagnostics. A result
supports only the changed modeling assumption, not a general ranking of
controllers.
:::
