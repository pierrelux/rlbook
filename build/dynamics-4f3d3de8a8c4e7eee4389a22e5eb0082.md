---
description: Formulate sequential decision problems by identifying their state, action channel, uncertainty, observations, constraints, and available model interface.
kernelspec:
  name: python3
  display_name: Python 3
---

# Dynamics and State-Space Models

A controller or learning algorithm acts on a description of a system. The
description may be an equation, a simulator, a collection of transitions, or an
interactive environment. Before choosing an algorithm, we need to determine
what evolves, what can be changed, what can be observed, and which physical and
operational restrictions must be preserved.

The worked systems isolate different parts of that formulation. SwingRL tests
whether a successful trajectory survives a change in suspension physics. A
three-station BIXI corridor separates mean flows from stochastic events and a
fixed schedule from inventory feedback. A camera gimbal separates sensor
readings from the hidden state needed for stabilization. The inference-serving
case then distinguishes equations, executable simulators, logged transitions,
and a live process without changing the system boundary.

::::{admonition} Learning Goals
:class: note

After reading this chapter, you should be able to:

- define the state, action, disturbance, observation, objective, and constraints of a sequential decision problem;
- locate the physical channel through which an action changes a system;
- write deterministic and stochastic dynamics in continuous- and discrete-time state-space form;
- distinguish an open-loop action sequence from a feedback policy;
- classify the information supplied by equations, simulators, transition samples, and logged data;
- represent partial observability with an observation model;
- separate known structure from components that may be learned, and use trajectory diagnostics to detect a missing physical mode or constraint.

**Prerequisites:** Linear algebra, multivariable calculus, elementary
probability, and the meaning of an ordinary differential equation. Numerical
integration is reviewed in [Solving Initial Value Problems](appendix_ivps.md).
::::

## System Boundaries and Action Channels

A playground swing, an overhead crane, and a wave-energy converter all
oscillate. Their visual similarity does not make them the same control problem.
A rider changes body shape, a crane accelerates the suspension point, and a
wave-energy converter changes a dissipative load while the sea supplies the
forcing.

:::{admonition} Formulation exercise
:class: tip
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

Many mechanical models can be organized around a generalized position $q$,
velocity $v$, auxiliary configuration $z$, and input $u$:

$$
\dot q=v,
\qquad
M(q,z)\dot v
=-\nabla V(q,z)-D(q,v,z)+B(q,v,z)u+d(t).
$$

The mass matrix $M$, potential $V$, dissipation $D$, actuation map $B$, and
disturbance $d$ have distinct physical roles. The term $B(q,v,z)u$ cannot be
specified from the word "action" alone. An unconstrained vector input would
omit the swing's internal-actuation geometry, the crane's acceleration limit,
and the wave device's passivity restriction.

The system boundary also determines which variables are external. A wave torque
is a disturbance for the energy converter but would be an action in a wave-tank
experiment. The boundary should follow the decision maker whose choices the
model is intended to support.

## Decision Models

In machine learning, a model often means a parameterized predictor. A
**decision model** contains more: a state, available actions, dynamics,
observations, an objective, constraints, and an information pattern. Prediction
remains one component, but decisions require counterfactual trajectories under
actions that may not yet have been observed.

For a finite horizon, one common formulation is

$$
\begin{aligned}
\underset{u_0,\ldots,u_{T-1}}{\operatorname{minimize}}\quad
& \mathbb{E}\!\left[\sum_{t=0}^{T-1}
  \ell_t(x_t,u_t)+\ell_T(x_T)\right] \\
\text{subject to}\quad
& x_{t+1}\sim P_t(\cdot\mid x_t,u_t), \\
& u_t\in\mathcal U_t(x_t),
\qquad x_t\in\mathcal X_t.
\end{aligned}
$$

This expression says what is optimized, how uncertainty propagates, and which
trajectories are admissible. Omitting one of these pieces changes the problem,
even if the same environment class and neural network are used afterward.

An **open-loop plan** selects the whole action sequence from information
available at the start,

$$
(u_0,\ldots,u_{T-1})=\mu(x_0).
$$

A **feedback policy** selects each action from the information available at that
time,

$$
u_t=\pi_t(I_t),
$$

where $I_t$ may contain the current state, observations, or a history. A planned
trajectory can perform well under its assumed initial condition and disturbance
forecast while reacting poorly to a tap, a delayed actuator, or an unexpected
arrival. Later chapters turn open-loop plans into feedback by replanning and by
computing state-contingent value functions.

Several independent choices describe the model itself.

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

## State-Space Models

A state is a summary of the past sufficient to predict the next state once the
current action and disturbance are specified. It is defined relative to a
model. A simulator may store a large internal state while exposing a smaller
observation to the controller.

In discrete time, deterministic state-space dynamics take the form

$$
x_{t+1}=f_t(x_t,u_t,d_t),
\qquad
y_t=h_t(x_t,u_t,v_t),
$$

where $d_t$ denotes a process disturbance and $v_t$ denotes measurement noise.
In continuous time, the corresponding equations are

$$
\dot x(t)=f(x(t),u(t),d(t)),
\qquad
y(t)=h(x(t),u(t),v(t)).
$$

Physical laws are often most legible in continuous time, while sensors,
actuators, data sets, and software interfaces operate at discrete times. A
numerical integrator and a sampling period connect the two descriptions. The
sampling period is part of the model because changing it changes both the
transition map and which disturbances can be resolved.

Linear continuous-time dynamics have the form

$$
\dot x=Ax+Bu+Ed,
\qquad
y=Cx+Du.
$$

The matrix $B$ describes how the action enters the dynamics, while $C$
describes what is measured. These matrices encode different questions. Full
actuation does not imply full observation, and full observation does not imply
that every state component can be controlled.

Linearity is a modeling choice rather than a claim that the world is linear at
all scales. A nonlinear system may be well approximated by a linear model near
an equilibrium. [Dynamic Programming](dp.md) uses that local approximation to
stabilize an inverted pendulum, while [Trajectory Optimization](trajectories.md)
uses the nonlinear equations to move the pendulum into the local region.

(internal-actuation-in-swingrl)=
## Internal Actuation and Unilateral Constraints in SwingRL

The [`swing-rl`](https://github.com/pierrelux/swing-rl) environment asks whether
a rider can pump a playground swing over the top bar without applying a motor
torque at the pivot. Its articulated standing model exposes two normalized
actions in \([-1,1]^2\): a squat target and a torso-lean target. The action
therefore changes body geometry. It is not a disguised generalized torque on
the suspension angle.

Let \(\theta\) be the suspension angle, \(\rho\) the rider's center-of-mass
distance from the pivot, \(\alpha\) its offset from the suspension line,
\(\psi=\theta+\alpha\), \(\beta\) the body orientation relative to the
suspension, and \(J\) the rider's inertia about its center of mass. SwingRL's
reduced equation is

$$
(m\rho^2+J)\ddot\theta
=-m\rho^2\ddot\alpha
-2m\rho\dot\rho\dot\psi
-J\ddot\beta
-\dot J(\dot\theta+\dot\beta)
-mg\rho\sin\psi
+\tau_{\mathrm{damp}}.
$$

Torso lean contributes through the driven terms in \(\ddot\alpha\) and
\(\ddot\beta\). Squatting changes \(\rho\) and contributes through the
parametric term \(-2m\rho\dot\rho\dot\psi\). Shortening near a bottom
passage adds energy in either direction because \(\dot\rho\) changes sign
with the phase of the swing.

The experiment audits a more basic assumption. It applies one phase-feedback
law to the standing body under two suspension models supplied by SwingRL. A
rigid rod can carry tension or compression. A chain can pull but cannot push.
The body parameters, initial state, squat and lean limits, controller mapping,
32-second horizon, and 0.02-second sampling interval remain fixed. A common
startup envelope

$$
r(t)=\min(t/0.25,1)
$$

ramps both action targets from neutral and removes an artificial release at the
first time step. Success requires a full \(2\pi\) crossing while the seat is
extended to at least \(0.9L\). Because this is feedback, the two action
sequences may diverge after the plant states diverge even though the controller
mapping is unchanged.

*Recorded SwingRL model audit.* The left plant permits bidirectional axial force;
the right plant enforces a unilateral chain. Prefix-only traces show unwrapped
angle, the rod's compression demand, and chain extension. Event buttons seek to
Python-recorded release, reattachment, snap, and rod-rotation times.

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

The rod completes a rotation, but part of that trajectory asks the suspension
to push. The chain instead enters a slack ballistic mode when its required
tension reaches zero. Reattachment at the chain-length boundary applies an
impact law and dissipates energy. This is a hybrid system because a parameter
change inside the rod equation cannot create the missing ballistic phase.

The conclusion is deliberately narrow: this controller passes the
bidirectional-rod model and does not solve the unilateral-chain plant. The
comparison neither ranks control against reinforcement learning nor shows that
no controller can rotate a chain swing. It shows why success on one executable
model does not validate the physical assumption that distinguished it from the
target plant.

A learned approximation fitted only to rod trajectories would inherit the same
blind spot. Data remain useful for estimating rider parameters, damping, and
contact losses after the unilateral mode structure is represented.
[Policy Gradients](pg.md) later evaluates PPO and the structured controller
against the same SwingRL action semantics and success criterion.

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

## Stochastic Dynamics

Deterministic dynamics assign one next state to each state-action pair. Process
noise, uncertain inflow, and unmodeled interactions instead produce a
distribution of possible next states. A constructive representation makes the
random input explicit:

$$
x_{t+1}=f_t(x_t,u_t,w_t),
\qquad
w_t\sim p_w.
$$

For additive Gaussian noise, this becomes

$$
x_{t+1}=Ax_t+Bu_t+w_t,
\qquad
w_t\sim\mathcal N(0,Q).
$$

The same stochastic dynamics can be represented directly by a transition
kernel,

$$
P_t(A\mid x,u)
=\Pr(x_{t+1}\in A\mid x_t=x,u_t=u),
$$

which assigns a probability to each measurable set of next states. The
function-plus-noise representation exposes how randomness enters and may permit
pathwise differentiation. The kernel representation requires only a
distribution over next states and includes simulators whose internal random
variables are hidden.

In continuous time, a stochastic differential equation separates drift and
diffusion:

$$
dX_t=f(X_t,U_t)\,dt+\sigma(X_t,U_t)\,dW_t.
$$

The sampling interval again matters. A discrete model obtained from this
equation must account for how diffusion accumulates over the interval rather
than reusing the same noise covariance at every resolution.

### Bicycle Inventory and Stochastic Demand

Between 07:00 and 10:00 on 4 July 2024, the BIXI stations at
Berri--Cherrier and Prince-Arthur--St-Urbain recorded many more completed
rentals than returns. The station at de Maisonneuve--Aylmer recorded the
opposite flow. The three stations are less than 1.6 km apart, so a small
relocation truck has a meaningful action without requiring a city-scale routing
model.

:::{figure} _static/bixi/bixi-model-interface.svg
:label: fig-bixi-model-interface
:alt: A coordinate-faithful schematic of three nearby BIXI stations and their completed rental and return counts during the recorded morning.

The three-station boundary contains 99 docks. Berri recorded 79 completed
rentals and 36 returns, Prince-Arthur recorded 48 and 20, and de Maisonneuve
recorded 12 and 84. The events come from the 2024 BIXI trip archive; capacities
come from a station-information snapshot retrieved on 2 September 2026.
:::

| station | ID | capacity |
|---|---:|---:|
| Berri / Cherrier | 173 | 39 |
| Prince-Arthur / St-Urbain | 404 | 23 |
| de Maisonneuve / Aylmer (ouest) | 68 | 37 |

Let \(s_{i,k}\) be the number of bicycles at station \(i\), \(c_i\) its
capacity, and \(b_k\) the truck inventory. Attempted rentals \(d_{i,k}\) and
returns \(a_{i,k}\) are disturbances. The numbers actually served are

$$
r_{i,k}=\min(d_{i,k},s_{i,k}),
\qquad
v_{i,k}=\min(a_{i,k},c_i-s_{i,k}).
$$

A transfer \(\rho_{i,k}\) is positive when the truck unloads bicycles at its
current station and negative when it loads them. The inventory balance is

$$
s_{i,k+1}=s_{i,k}-r_{i,k}+v_{i,k}+\rho_{i,k},
\qquad
b_{k+1}=b_k-\sum_i\rho_{i,k}.
$$

Relocation cancels when station and truck inventories are added. Customer trips
can still change the total inside this deliberately small boundary because
their other endpoint may lie elsewhere in the network.

The teaching plant begins with station inventories \((30,18,8)\) and an empty
16-bike truck at de Maisonneuve. Decisions occur every 15 minutes. A stop may
transfer at most eight bicycles, and travel between any two of the three
stations consumes one decision interval. The simulator processes returns before
rentals when timestamps coincide and records lost rentals and rejected returns
instead of silently clipping the requested flow.

Three controllers receive the same event trace. The first never relocates a
bike. The second freezes a schedule computed before 07:00 from mean completed
trip counts. The third applies the same routing rule every 15 minutes to the
observed inventories. Both active controllers use the remaining-horizon target

$$
g_{i,n}=
\operatorname{clip}\left(
\frac{c_i}{2}
+\sum_{h=n}^{T-1}(\hat d_{i,h}-\hat a_{i,h}),
0.2c_i,0.8c_i
\right).
$$

The truck loads only when its current station has a surplus of at least two
bicycles and another station has a deficit of at least two. It unloads under
the symmetric deficit condition, then routes toward the largest normalized
deficit or surplus. The frozen mean-flow schedule waits at de Maisonneuve,
loads three bicycles at 07:45, travels to Prince-Arthur, and unloads them at
08:00. Feedback may change that route after observing new events.

*Recorded BIXI controller comparison.* The map, station fills, truck motion,
inventory traces, and service failures come from committed Python trajectories.
Controller selection and the playhead only seek through those records; they do
not recompute the plant in the browser.

```{code-cell} python
:tags: [remove-input]

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from bixi_replay import render_bixi_replay

display(HTML(render_bixi_replay(
    Path("artifacts/bixi/textbook_results.json"),
    replay_id="bixi-modeling-replay",
    fallback_id="fig-bixi-control-replay-fallback",
)))
```

:::{figure} _static/bixi/bixi-feedback-evidence.svg
:label: fig-bixi-control-replay-fallback
:class: pdf-fallback
:alt: BIXI station inventory trajectories and paired service-failure intervals for no relocation, a frozen schedule, and inventory feedback.

Static trajectories and paired stochastic evidence from the same Python
experiment. The online book adds controller selection, playback, stepping, and
scrubbing.
:::

Completed-trip means from 43 earlier weekdays parameterize independent Poisson
event counts. Seeds 0 through 511 use common random numbers across controllers.
A second, declared disturbance adds eight unexpected rental attempts at
Prince-Arthur at 07:47 and eight independent return attempts at de Maisonneuve
at 07:57. The feedback controller first observes their effect at 08:00; the
frozen schedule does not.

```{include} artifacts/bixi/results.md
```

Feedback reduces service failures in this model, but it does more relocation
work. These numbers do not estimate the causal effect of BIXI's operations.
The public archive contains completed trips, not unsuccessful attempts,
historical inventories, or operator truck movements. The Poisson model is a
transparent distribution calibrated to completed events rather than a claim
about latent demand. The selected morning was chosen because its imbalance is
visually legible {cite}`bixiOpenData2024,montrealBixiTrips,hulot2018bixi`.

:::{dropdown} Inspect the BIXI transition and feedback rule
```{literalinclude} code/bixi_control.py
:language: python
:start-at: class InventoryFeedbackController
:end-before: def _sha256
:linenos:
```

```{literalinclude} code/bixi_control.py
:language: python
:start-at: def simulate(
:end-before: def simulate_fluid
:linenos:
```

{download}`Download the complete BIXI experiment <code/bixi_control.py>`
:::

A hydroelectric reservoir has the same accounting pattern at a larger scale.
With storage \(x_t\), release \(u_t\), spill \(s_t\), and uncertain inflow
\(w_t\), its balance is \(x_{t+1}=x_t+w_t-u_t-s_t\). Historical data may supply
an inflow distribution without replacing water conservation or the reservoir's
capacity limits.

## Partial Observability

The state need not be directly measured. An observation model relates hidden
state to sensor output:

$$
y_t=h(x_t,u_t,v_t),
\qquad
v_t\sim p_v.
$$

For a linear Gaussian sensor, $y_t=Cx_t+v_t$ with
$v_t\sim\mathcal N(0,R)$. A controller based only on $y_t$ may lose information
needed to predict future transitions. An estimator can instead summarize the
observation history as a state estimate or a distribution over possible states.

### Camera Stabilization: A Measurement Is Not a State

Consider a balanced camera mounted on a one-axis gimbal. Its hidden state

$$
x=(\theta,\omega,b)
$$

contains the camera angle, angular velocity, and gyroscope bias. The mechanical
plant is

$$
\dot\theta=\omega,
\qquad
J\dot\omega=u-c\omega+d_\tau(t),
\qquad |u|\leq u_{\max}.
$$

The motor does not receive this state directly. A gyroscope measures angular
velocity plus a slowly varying bias,

$$
y_k^\omega=\omega_k+b_k+\nu_k^\omega.
$$

An accelerometer supplies a gravity reference only when translational
acceleration is negligible. In the simulated plane,

$$
\begin{bmatrix}y_k^x\\y_k^y\end{bmatrix}
=R(-\theta_k)
\begin{bmatrix}a_{x,k}^{w}\\g\end{bmatrix}+\nu_k^a.
$$

Without measurement noise, the apparent tilt is

$$
\operatorname{atan2}(y_k^x,y_k^y)
=\theta_k+\operatorname{atan2}(a_{x,k}^{w},g).
$$

The same sensor output can therefore result from camera tilt, base
acceleration, or a combination of both. Integrating the gyroscope avoids that
short-term ambiguity, but an unknown bias then accumulates as angle error.

The experiment changes only the estimator. Every run receives the same plant,
noise realization, disturbances, torque limit, and saturated feedback law

$$
u_k=\operatorname{clip}
\left(-0.9\hat\theta_k-0.12\hat\omega_k,
-0.18,0.18\right).
$$

The teaching plant uses (J=0.018\ \mathrm{kg\,m^2}),
(c=0.025\ \mathrm{N\,m\,s/rad}), one-millisecond integration, and
ten-millisecond sensing and control. The ten-second sensor trace is generated
once with seed 11 and then supplied unchanged to every estimator.

All three controllers first recover from an eight-degree tilt and reject a
(0.14\ \mathrm{N\,m}) mechanical tap from 1.50 to 1.62 seconds. A
(3\ \mathrm{m/s^2}) base-translation pulse from 4.0 to 4.6 seconds then tests whether the
estimated state confuses translation with rotation. The final interval exposes
the drift caused by integrating a biased gyroscope.

*Recorded gimbal comparison.* The plant, feedback law, disturbance sequence,
torque limit, and sensor-noise realization remain fixed. Curves reveal only
samples reached by the playhead, and the metrics use the hidden simulated
state.

```{code-cell} python
:tags: [remove-input]

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from gimbal_replay import render_gimbal_replay

display(HTML(render_gimbal_replay(
    Path("artifacts/gimbal/textbook_results.json"),
    replay_id="gimbal-observation-replay",
    fallback_id="fig-gimbal-observation-fallback",
)))
```

:::{figure} _static/gimbal/partial-observability.svg
:label: fig-gimbal-observation-fallback
:class: pdf-fallback
:alt: Three camera gimbals and their true-angle trajectories under accelerometer-only, integrated-gyro, and complementary state estimation.

Static snapshots and complete trajectories from the same Python experiment.
The online book adds playback, stepping, and scrubbing.
:::

```{include} artifacts/gimbal/results.md
```

No sensor is declared useless by this comparison. The accelerometer supplies a
long-run reference but responds to specific force, while the gyroscope supplies
short-run angular-rate information but carries an unknown bias. A complementary
observer combines these signals and estimates the bias
{cite}`androidMotionSensors2026,mahony2008complementary`. It reduces both tested
failure modes, but sustained unknown translation remains confounded with
gravity. These are results from a transparent teaching simulation, not measured
performance of a commercial gimbal.

The same distinction appears in adaptive optics: local wavefront slopes are
sensor observations, while the reconstructed phase is the state used for
control.

:::{dropdown} Inspect the sampled plant, sensors, estimator, and controller
```{literalinclude} code/gimbal_control.py
:language: python
:start-at: def sample_observation
:end-before: def compute_metrics
:linenos:
```

{download}`Download the complete gimbal experiment <code/gimbal_control.py>`
:::

## Model Interfaces

Two models may generate the same nominal trajectory while exposing different
operations to an algorithm. Access to equations, derivatives, resets, and new
interactions determines what can be computed.

| available interface | operations it supports directly | methods developed later |
|---|---|---|
| equations and derivatives | evaluate local dynamics, linearize, differentiate constraints | direct transcription, LQR, gradient-based MPC |
| one-step transition function | reset and advance from chosen state-action pairs | shooting, simulation-based MPC, model-based dynamic programming |
| generative simulator | sample trajectories without inspecting internal equations | Monte Carlo estimation, derivative-free search, policy learning |
| logged transitions | fit or evaluate models on the recorded distribution | system identification, fitted value and Q methods, offline evaluation |
| interactive environment and objective | collect new transitions under chosen actions | online reinforcement learning |

The rows are not mutually exclusive. A simulator may be differentiable, an
explicit model may contain unknown parameters, and logged data can be used to
fit a new one-step model. The table records the information supplied to the
algorithm, not the origin or fidelity of the model.

The inference case can be accessed through four of these interfaces without
changing the system boundary. Token, cache, and thermal balances supply
equations. The profile-based event simulator permits resets and counterfactual
clock or scheduling actions. The trace and profiled transitions support only
inferences on their recorded distribution. A live Qwen/vLLM process permits
new requests under chosen clock settings. Moving down this list removes
operations from the algorithm; it does not turn the underlying serving process
into a different system.

### Inference Serving as a Controlled System

An inference server receives prompts at irregular times and generates one
response for each request. A decoder-only language model processes a request in
two phases. The **prefill** phase processes the prompt in parallel and creates
the first output token. The **decode** phase generates the remaining tokens one
iteration at a time. Long prefills can delay the short decode iterations of
requests already in progress. Iteration-level scheduling and chunked prefill
are production approaches to controlling this interaction
{cite}`yu2022orca,agrawal2024sarathi`. The teaching simulator below uses a
reduced interleaving rule rather than reproducing either serving system.

The system boundary surrounds one GPU and its serving process. The language
model and inference engine lie inside the boundary; arriving requests and the
ambient thermal conditions lie outside it. At a one-second control interval, a
useful aggregate state is

$$
x_t=(p_t,d_t,m_t,T_t,f_t),
$$

where $p_t$ is queued prefill work, $d_t$ is unfinished decode work, $m_t$ is
key-value-cache occupancy, $T_t$ is GPU temperature, and $f_t$ is the realized
graphics clock. The control

$$
u_t=(f_t^{\mathrm{req}},\sigma_t)
$$

combines a requested clock with a scheduling rule $\sigma_t$. The scheduler
decides which phase receives each service step and how much prefill work may be
processed before returning to active decodes. The clock request changes the
rate and energy cost of that work. Hardware may realize a lower clock under a
power or thermal limit,
so $f_t^{\mathrm{req}}$ and $f_t$ are distinct variables
{cite}`nvidiaSmi2026`.

Request arrivals form the disturbance. An arrival supplies its time and prompt
length, but its eventual output length remains unknown to the controller until
the end-of-sequence token arrives. The observation contains queue ages,
completed prompt and output tokens, cache use, power, temperature, and realized
clock. This information pattern prevents a controller from scheduling with the
future length recorded in an evaluation trace.

The aggregate balances expose structure that does not have to be learned. If
$A_t^p$ prompt tokens arrive, $C_t^p$ prompt tokens are processed, $N_t^p$
requests finish prefill, and $C_t^d$ decode tokens are produced, then

$$
\begin{aligned}
p_{t+1} &= p_t+A_t^p-C_t^p,\\
d_{t+1} &= d_t+W_t^d(N_t^p)-C_t^d.
\end{aligned}
$$

The term $W_t^d(N_t^p)$ denotes the still-unknown output work associated with
newly admitted decode requests. The request-level simulator reveals this work
only as tokens complete. It tracks continuous prompt and generated-token
occupancy, including partial prefills, together with a fixed per-request
reserve. That occupancy is released when a request completes. This aggregate
accounting preserves the modeled cache balance, but it does not reproduce a
serving engine's block allocator. An aggregate thermal balance has the form

$$
T_{t+1}=T_t+\frac{\Delta t}{C_\theta}
\left(P(x_t,u_t)-\frac{T_t-T_{\mathrm{amb}}}{R_\theta}\right).
$$

Token and cache conservation determine the form of the transition. Measurements
are still needed for the service-rate map, the power map $P$, and the thermal
parameters. The profiling protocol targets Qwen2.5-7B-Instruct served by vLLM
on an NVIDIA L4 {cite}`qwen25report,kwon2023vllm`. Its intended hardware profile
spans five requested clock levels, several prompt lengths, and three concurrency
levels.
The book build reads a committed profile and never starts a model server. Its
manifest states whether the maps come from completed L4 measurements or a
pre-measurement engineering surrogate, and every rendered result displays that
provenance.

The workload is a five-minute excerpt from the Azure 2023 code-generation
trace, which records request times and input and output token counts
{cite}`azureLLMTrace2023,patel2024splitwise`. Arrival times are dilated once to
place maximum-clock utilization near 80 percent. If $\rho_{\max}$ is the
isolated service time of all requests at the highest profiled clock divided by
the original window length, the dilation and normalized arrivals are

$$
d=\max\!\left(1,\frac{\rho_{\max}}{0.8}\right),
\qquad
t_i'=d(t_i-t_0).
$$

All controllers receive the same immutable requests after this transformation.
The experimental time-to-first-token limit is twice the median baseline value
for a 1,024-token prompt at concurrency one and the highest clock. The
time-per-output-token limit is 1.5 times the corresponding median. These are
reference levels for a controlled comparison, not service-level objectives
claimed for production systems.

| quantity | experiment setting |
|---|---|
| model | Qwen2.5-7B-Instruct, revision `acbd96531cda22292a3ceaa67e984955d3965282` {cite}`qwen25modelcard` |
| inference engine | vLLM OpenAI server, version 0.28.0 {cite}`vllmDocker028` |
| target accelerator | one NVIDIA L4 |
| clock profile | five requested graphics clocks; realized clock, power, utilization, and temperature retained |
| request trace | first five minutes for evaluation; first 20 requests for the animation |
| controller sampling | one second |
| service simulation | 0.1-second steps; decode priority or one prefill chunk first; unused prefill capacity may return to decode |
| output length | hidden from the controller until completion |

Within each one-second clock-control period, the scheduler gives each 0.1-second
simulator step to decode or begins it with one prefill chunk. If that chunk
finishes before the step's service budget is exhausted, the remaining capacity
returns to decode. When both phases have work, active decode receives
alternating-step priority and strict priority under high cache pressure. This
**512-token interleaved chunked-prefill** rule keeps the action channel visible
without claiming to reproduce vLLM's mixed-batch scheduler or Sarathi-Serve.

The first 20 requests form the animation below.

*Recorded inference-serving model.* Twenty requests from the Azure
code-generation trace move through arrival, interleaved 512-token prefill
chunks, autoregressive decode, and completion. The plots report only the
trajectory prefix reached by the playhead, and output length is hidden from the
controller until each request completes. A provenance badge distinguishes a
verified measured profile from a pre-measurement engineering surrogate.

```{code-cell} python
:tags: [remove-input]

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from inference_replay import render_serving_replay

display(HTML(render_serving_replay(
    Path("artifacts/inference_serving/textbook_results.json"),
    view="modeling",
)))
```

:::{figure} _static/inference_serving/modeling.svg
:label: fig-inference-serving-model-fallback
:class: pdf-fallback
:alt: Line charts of unfinished requests and realized GPU clock for the 20-request modeling replay.

Static summary of unfinished requests and realized clock from the same
request-level simulation. The online book provides playback, stepping,
scrubbing, phase detail, and controller selection.
:::

The experiment asks whether this state is sufficient to reproduce the service
phenomena that matter for control. Conservation checks account for every
request, processed token, and cache allocation.
A concrete accounting check is visible at the end of the committed replay: all
20 requests have completed, the prefill and decode queues are empty, and all
modeled cache occupancy has been released. This establishes closure of the modeled
flows. It does not establish that the service-rate, power, or thermal maps are
accurate.
A provisional profile permits tests of the simulation and control software but
does not supply empirical latency or energy evidence. In either case, the model
does not cover every inference engine or GPU. Network transfer, host-side
tokenization, multi-GPU communication, model-quality effects, and failures
outside the serving process remain outside the boundary.

:::{dropdown} Inspect the request-level transition
```{literalinclude} code/inference_serving.py
:language: python
:start-at: def simulate
:end-before: def compute_metrics
:linenos:
```

{download}`Download the complete inference-serving model <code/inference_serving.py>`
:::

### Completed Trips Are Not Attempted Demand

The BIXI archive supplies a different interface: a log of completed trips. A
completed rental proves that a bicycle was available, but an empty station
leaves no trip record for a customer who could not depart. A completed return
similarly proves that a dock was available without counting customers who found
the station full.

:::{figure} _static/bixi/bixi-completed-trip-censoring.svg
:label: fig-bixi-completed-trip-censoring
:alt: Two attempted-demand histories produce the same BIXI completed-trip log because extra attempts occur while a station is empty.

Two worlds can produce the same completed-trip log. In the second world,
additional customers attempt to rent while the station is empty. The log
cannot identify those censored attempts.
:::

The public station-status feed can reveal current inventories while it is
running, but it does not provide an action channel for relocating bicycles.
Historical completed trips omit unsuccessful demand, past station inventories,
and operator movements. A fitted arrival model can be useful, but evaluating a
new relocation policy from these logs requires assumptions about the missing
attempts and about how operations generated the observed data. The executable
BIXI example keeps those assumptions visible by declaring an attempted-demand
model and recording every rejected event.

:::{dropdown} Inspect the logged-data counterexample
```{literalinclude} code/bixi_control.py
:language: python
:start-at: def make_censoring_counterexample
:linenos:
```
:::


A program is therefore a model when its execution defines state transitions.
MuJoCo combines rigid-body equations, contact detection, constraint forces,
sensors, and rendering. A user may have a reset-and-step interface without
having a practical expression for its complete local transition function.
Discrete-event simulators instead advance from one asynchronous event to the
next. The inference-serving simulator advances through arrivals, completed
prefill chunks, decode iterations, and request departures. It still defines a
transition model, although its event clock and variable population of active
requests differ from the fixed-step state-space models above.

Calling a method **model-free** describes an information restriction. The
method does not query an explicit transition model during its update. It still
depends on a specified state or observation, action set, reward, sampling
process, and assumptions about how experience relates to future deployment.

### Language Generation as a Sequential System

Autoregressive language generation fits the same notation with an unusual
transition. For a fixed prompt $p$, let

$$
s_t=(p,y_1,\ldots,y_{t-1}),
\qquad
a_t=y_t.
$$

The base transition appends the selected token,

$$
s_{t+1}=\operatorname{concat}(s_t,a_t).
$$

This transition is deterministic once the token is chosen; uncertainty comes
from the token policy $\pi(a_t\mid s_t)$. A language model therefore supplies a
policy, while the prefix update supplies the transition. A reward or preference
model is an additional object rather than an inherent property of text
generation.

The simplified boundary changes when the model calls tools, receives new user
messages, or interacts with an external application. Tool outputs and user
responses then become observations generated by an environment outside the
prefix concatenation rule. Context truncation also requires the retained
context or memory state to be specified explicitly.

## Known Structure and Learned Components

Uncertainty in one component does not require replacing the entire model. A
structured transition can retain known dynamics and add a learned residual:

$$
x_{t+1}=f_{\mathrm{known}}(x_t,u_t)
+r_\theta(x_t,u_t,z_t)+w_t.
$$

The feature vector $z_t$ may contain weather, payload information, or other
measured conditions. The residual $r_\theta$ has a narrower task than a complete
black-box transition: it predicts the part left unexplained by the known model.

| system | retained structure | plausible learned component | model-audit experiment |
|---|---|---|---|
| swing | internal actuation and unilateral tension | rider, damping, and contact-loss parameters | measure the first slack event |
| bicycle corridor | inventory and truck conservation, dock and truck capacities | attempted-demand distribution and travel times | introduce a declared demand shock |
| camera gimbal | rigid-body dynamics, torque limit, and sensor geometry | gyro-bias evolution or disturbance model | separate mechanical rotation from base translation |
| inference service | request, token, and cache conservation | service-rate, power, and thermal residuals | hold out prompt lengths, clocks, and concurrency levels |

Generative models can produce plausible simulator and controller code quickly.
That code does not establish that a chain can sustain the simulated force, that
an action has the intended physical meaning, or that a scalar reward contains a
safety requirement. Those claims require inspection of the model and evidence
from targeted experiments.

A residual cannot repair a missing action channel or a missing physical mode if
the training data never contains evidence of it. The SwingRL comparison exposed
the failure by changing the suspension model while holding the controller fixed.
Comparable interventions are needed to decide which structure should remain and
which component should be learned.

## Summary

A decision model specifies state, action, dynamics, observations, objective,
constraints, time, and information. The state-space representation separates
what evolves from what is observed and what can be controlled. Continuous and
discrete time, deterministic and stochastic evolution, and full and partial
observation are independent modeling choices.

The SwingRL experiment asks whether success survives a change in physical
assumptions. The BIXI experiment separates stochastic customer events from the
truck's constrained action and compares a frozen schedule with inventory
feedback. The gimbal experiment separates noisy measurements from a predictive
state estimate. Inference serving then shows how the same system boundary can
be exposed through equations, a simulator, logs, or a live process.

The model interface determines the methods available to the decision maker.
The next chapter assumes a dynamics model that can generate and constrain
candidate trajectories, then computes an open-loop plan. Later chapters add
feedback and learn model components, values, or policies when the corresponding
objects are unavailable or too costly to compute directly.

```{admonition} Exercises
:class: hint dropdown

1. For an unfamiliar system, write five lines specifying its state, action,
   disturbance, objective, and one hard constraint. State where the system
   boundary is drawn.

2. In the action-channel figure, explain why replacing every input by an
   unconstrained generalized force changes all three physical problems.

3. Convert $\dot x=-ax+bu$ to a discrete-time model using forward Euler with
   step size $\Delta t$. Identify the resulting coefficients.

4. Use the sign of $-2m\rho\dot\rho\dot\psi$ to explain why shortening the
   rider's radius near the bottom can add energy on both half-cycles.

5. Express the swing's taut and slack phases as a hybrid system. Identify the
   continuous state, discrete mode, release guard, and reattachment event.

6. The structured swing controller succeeds on the rod and fails on the chain.
   Explain why fitting a larger neural transition model to rod-only data does
   not, by itself, resolve this discrepancy.

7. For the BIXI corridor, verify that relocation preserves the sum of station
   and truck inventories. Explain why customer trips need not preserve that sum
   inside the three-station boundary.

8. Construct two pairs \((\theta,a_x^w)\) that yield the same noiseless
   accelerometer-derived tilt. Explain which additional information the
   gyroscope supplies and which ambiguity remains.

9. For the inference service, write a cache-balance equation that distinguishes
   blocks allocated at prefill completion from blocks released at request
   completion. State one condition that makes the aggregate cache state
   insufficient for prediction.

10. Write the reservoir transition as a function-plus-noise model when inflow
    is $w_t=\bar w_t+\epsilon_t$. Describe the corresponding transition kernel.

11. Give an example where the observation is not a Markov state. State which
   missing variable or history would improve prediction.

12. Classify each of the following interfaces: a differentiable ODE solver, a
    reset-and-step robotics simulator, and a fixed table of logged transitions.
    Name one method from the book that each interface supports directly.

13. For autoregressive language generation, distinguish the environment
    transition from the token policy. Explain how the model changes when a tool
    call returns information from an external database.

14. Extend the SwingRL experiment with the seated controller, or disable either
    squat or lean in the standing controller. Predict the diagnostic that will
    change before running the code. Explain which modeling assumption the
    intervention tests.

15. Starting from the completed-trip censoring counterexample, add one failed
    rental attempt while the station is empty. Explain why this changes latent
    demand without changing the completed-trip log.

16. Load the Azure trace subset used by the inference experiment. Recompute the
    offered-load dilation after omitting the ten longest prompts. Explain which
    model quantity changes and which conservation equations remain unchanged.
```

## Computational Sources

The SwingRL dependency is pinned to commit
[`d579663`](https://github.com/pierrelux/swing-rl/commit/d579663fc81c044729f4d3ab60bf63bcdbd27b9a).
The chapter executes domain code and reads committed experiment artifacts:

- {download}`SwingRL controller and model audit <code/swing_control.py>`
- {download}`BIXI inventory model and controllers <code/bixi_control.py>`
- {download}`BIXI recorded replay renderer <code/bixi_replay.py>`
- {download}`Camera-gimbal plant, estimators, and controller <code/gimbal_control.py>`
- {download}`Camera-gimbal recorded replay renderer <code/gimbal_replay.py>`
- {download}`Action-channel figure <code/modeling_interfaces.py>`
- {download}`Inference-serving model <code/inference_serving.py>`
- {download}`Inference replay renderer <code/inference_replay.py>`

The animation controls are embedded in the static site and require no live
Python kernel.

## Self-checks

:::{exercise} State or observation?
:label: ex-dynamics-check-1

An inference controller observes queue counts and cache occupancy but not the
remaining output length of each active request. Are those measurements alone a
Markov state for the request-level process? Explain briefly.
:::

:::{solution} ex-dynamics-check-1
:class: dropdown

Generally no. Two systems can have the same counts and cache occupancy while
their active requests have different remaining output lengths, which changes
their future completion times and cache release.
:::

:::{exercise} Discretization check
:label: ex-dynamics-check-2

For $\dot{x}=ax+bu$ with a zero-order-held input and step $\Delta t$, write the explicit Euler update.
:::

:::{solution} ex-dynamics-check-2
:class: dropdown

$x_{k+1}=x_k+\Delta t(ax_k+bu_k)=(1+a\Delta t)x_k+b\Delta t\,u_k$.
:::

:::{exercise} Where uncertainty belongs
:label: ex-dynamics-check-3

When is a transition kernel more natural than writing deterministic dynamics plus additive noise?
:::

:::{solution} ex-dynamics-check-3
:class: dropdown

A kernel is more natural when uncertainty is discrete, state dependent, multimodal, or otherwise cannot be represented faithfully as a simple additive disturbance.
:::
