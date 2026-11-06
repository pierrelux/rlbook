---
description: Formulate sequential decision problems by identifying their state, action channel, uncertainty, observations, constraints, and available model interface.
kernelspec:
  name: python3
  display_name: Python 3
---
# Stochastic Dynamics and Partial Observation

Deterministic dynamics assign one next state to each state-action pair. Process
noise, uncertain inflow, and unmodeled interactions instead produce a
distribution of possible next states. This distinction changes the decision
problem: a controller may need to compare expected cost, failure probability,
or risk across those possible outcomes rather than optimize one nominal
trajectory.

How should prediction and action selection be defined when the next state is a
distribution and the controller may not observe the realized state directly?

One representation makes the source of randomness explicit:

$$
x_{t+1}=f_t(x_t,u_t,\xi_t),
\qquad
\xi_t\mid x_t,u_t\sim p_t(\cdot\mid x_t,u_t).
$$

For a realized disturbance $\xi_t$, the function $f_t$ still returns one next
state. The conditional law of $\xi_t$ determines how likely the different
realizations are and therefore induces a distribution over $x_{t+1}$. The
familiar additive Gaussian model assumes an independent noise sequence with a
fixed covariance:

$$
x_{t+1}=Ax_t+Bu_t+\xi_t,
\qquad
\xi_t\sim\mathcal N(0,Q).
$$

The induced dynamics can also be represented directly by a transition kernel,

$$
P_t(A\mid x,u)
=\Pr(x_{t+1}\in A\mid x_t=x,u_t=u).
$$

For any set $A$ of possible next states, $P_t(A\mid x,u)$ is the probability
that the next state lies in $A$ after action $u$ is applied in state $x$. This
object is called a **transition kernel**. The
function-plus-noise representation exposes how randomness enters and may permit
pathwise differentiation, in which a sampled transition is differentiated while
holding its random draw fixed. The kernel requires only the conditional distribution
of the next state, so it also covers simulators whose internal random variables
are hidden.

Continuous-time noise must be scaled consistently with the duration of a time
interval. A stochastic differential equation separates the deterministic rate
of change, called the drift, from rapidly fluctuating random increments, called
the diffusion:

$$
dX_t=f(X_t,U_t)\,dt+\sigma(X_t,U_t)\,dW_t.
$$

Here $f$ is the drift, $W_t$ is standard Brownian motion, and $\sigma$ maps its
increments into the state. Over an interval of length $\Delta t$, Brownian
motion satisfies

$$
W_{t+\Delta t}-W_t\sim\mathcal N(0,\Delta t I).
$$

The increment variance is proportional to elapsed time. The stochastic analogue
of a forward Euler step, called Euler--Maruyama, therefore has the form

$$
X_{k+1}\approx X_k+f(X_k,U_k)\Delta t
+\sigma(X_k,U_k)\sqrt{\Delta t}\,\varepsilon_k,
\qquad \varepsilon_k\sim\mathcal N(0,I).
$$

The random increment's standard deviation therefore scales as
$\sqrt{\Delta t}$ and its covariance as $\Delta t$. Halving the sampling
interval does not mean adding the same covariance twice as often. A full
treatment of stochastic differential equations lies outside this book; these
relations record the scaling needed to construct a consistent sampled model.

### Bicycle Inventory and Stochastic Demand

Inventory provides a concrete transition model in which random events and hard
constraints interact. A rental can occur only when a bicycle is present, and a
return can occur only when a dock is open. Relocation decisions change those
conditions before the next random arrivals are realized.

Between 07:00 and 10:00 on 4 July 2024, the BIXI stations at
Berri--Cherrier and Prince-Arthur--St-Urbain recorded many more completed
rentals than returns. The station at de Maisonneuve--Aylmer recorded the
opposite flow. The three stations are less than 1.6 km apart, so a small
relocation truck can move bicycles among them within one 15-minute interval.
This deliberately local boundary lets us study inventory feedback without also
having to choose routes across the full Montreal network.

:::{figure} _static/bixi/bixi-model-interface.svg
:label: fig-bixi-model-interface
:alt: A coordinate-faithful schematic of three nearby BIXI stations and their completed rental and return counts during the recorded morning.

The three-station boundary contains 99 docks. Berri recorded 79 completed
rentals and 36 returns, Prince-Arthur recorded 48 and 20, and de Maisonneuve
recorded 12 and 84. The events come from the 2024 BIXI trip archive; capacities
come from a station-information snapshot retrieved on 2 September 2026.
:::

The simulation uses the station capacities recorded in a 2 September 2026
station-information snapshot for every run. Those values make the example
reproducible, but they need not equal the capacities in service on 4 July 2024.

| station | ID | capacity |
|---|---:|---:|
| Berri / Cherrier | 173 | 39 |
| Prince-Arthur / St-Urbain | 404 | 23 |
| de Maisonneuve / Aylmer (ouest) | 68 | 37 |

Let $s_{i,k}$ be the number of bicycles at station $i$, $c_i$ its
capacity, and $b_k$ the truck inventory. A transfer $\rho_{i,k}$ is positive
when the truck unloads bicycles at its current station and negative when it
loads them. The post-transfer stock at the start of an interval is
$q_{i,0}=s_{i,k}+\rho_{i,k}$.

Attempted rentals and returns then arrive as a chronological event sequence.
For event $e$, let $R_{i,e}$ indicate a served rental and $A_{i,e}$ an accepted
return. Their values depend on the stock immediately before the event:

$$
\begin{aligned}
R_{i,e}
&=\mathbf 1\{\text{event $e$ is a rental at $i$}\}
  \mathbf 1\{q_{i,e-1}\geq 1\},\\
A_{i,e}
&=\mathbf 1\{\text{event $e$ is a return at $i$}\}
  \mathbf 1\{q_{i,e-1}<c_i\},\\
q_{i,e}&=q_{i,e-1}-R_{i,e}+A_{i,e}.
\end{aligned}
$$

If interval $k$ contains $N_k$ events, its inventory balance is

$$
s_{i,k+1}=q_{i,N_k},
\qquad
b_{k+1}=b_k-\sum_i\rho_{i,k}.
$$

Relocation cancels when station and truck inventories are added. Customer trips
can still change the total inside this three-station boundary because
their other endpoint may lie elsewhere in the network.

The teaching plant begins with station inventories $(30,18,8)$ and an empty
16-bike truck at de Maisonneuve. Decisions occur every 15 minutes. A stop may
transfer at most eight bicycles, and travel between any two of the three
stations consumes one decision interval. At a common station and timestamp,
the simulator processes returns before rentals. It records lost rentals and
rejected returns instead of silently clipping attempted events.

Three controllers receive the same event trace. The first never relocates a
bike. The second freezes a schedule computed before 07:00 from mean completed
trip counts. The third reevaluates a deterministic, hand-written rule every 15
minutes using the observed inventories. It does not solve an optimization
problem at each decision time. Here *feedback* means that the current
observation enters the rule; it does not imply optimization or learning.

Both active controllers use the remaining-horizon target

$$
g_{i,n}=
\operatorname{clip}\left(
\frac{c_i}{2}
+\sum_{h=n}^{T-1}(\hat d_{i,h}-\hat a_{i,h}),
0.2c_i,0.8c_i
\right).
$$

The function $\operatorname{clip}(q,l,r)$ limits $q$ to the interval $[l,r]$.
Thus the target begins at half capacity, shifts upward when future departures
are expected to exceed arrivals, and remains between 20 and 80 percent of
station capacity. Here $\hat d_{i,h}$ and $\hat a_{i,h}$ are the mean numbers of completed
departures and arrivals at station $i$ during 15-minute interval $h$ across the
43 calibration dates. Define the current deficit and surplus relative to the
target by

$$
\delta^-_{i,n}=[g_{i,n}-s_{i,n}]_+,
\qquad
\delta^+_{i,n}=[s_{i,n}-g_{i,n}]_+.
$$

The positive-part operator $[q]_+=\max(q,0)$ makes deficit and surplus
nonnegative. At most one is positive for a given station. The rule first
decides what to transfer at the truck's current station. If the
truck carries bicycles and the local deficit is at least two, it unloads. If
the local surplus is at least two and the other stations have a combined
deficit of at least two, it loads. Transfer size is capped by the eight-bike
stop limit, truck capacity, available bicycles, open docks, and the relevant
surplus or deficit. If the truck still carries bicycles after the transfer, it
drives toward the other station with the largest ratio
$\delta^-_{i,n}/c_i$. If it is empty, and both a deficit and a surplus of at
least two remain, it drives toward the other station with the largest ratio
$\delta^+_{i,n}/c_i$. Distance breaks ties. Finally, when the
truck carries leftover bicycles but the total modeled deficit is below two, it
returns as many as possible at its current station rather than keeping them
unavailable to customers.

The frozen schedule is produced before 07:00 by applying this same heuristic to
one deterministic mean-flow trajectory. It then keeps those actions unchanged.
That calculation makes it an open-loop plan, not a second online optimizer. In
the resulting schedule, the truck waits at de Maisonneuve, loads three bicycles
at 07:45, travels to Prince-Arthur, and unloads them at 08:00. The feedback
controller instead recomputes the rule from each newly observed inventory.

For the stochastic comparison, let $C^z_{i,k,d}$ be the number of completed
events of type $z\in\{\mathrm{rental},\mathrm{return}\}$ at station $i$ in
interval $k$ on calibration date $d$. The empirical rate and simulated count
are

$$
\widehat\lambda^z_{i,k}
=\frac{1}{43}\sum_{d=1}^{43}C^z_{i,k,d},
\qquad
N^z_{i,k}\sim
\operatorname{Poisson}\!\left(\widehat\lambda^z_{i,k}\right).
$$

The Poisson distribution is a count model whose mean here is
$\widehat\lambda^z_{i,k}$. The count draws are independent across stations,
intervals, and event types.
Conditional on $N^z_{i,k}=m$, the simulator places the $m$ event times
independently and uniformly inside interval $k$. It treats these events as
potential attempts and accepts or rejects each one using the inventory
equations above. Thus the model turns rates estimated from *completed* trips
into attempted-event rates. This is a declared modeling assumption: the public
archive does not identify demand that was censored by an empty or full station.

For each seed from 0 through 511, all three controllers receive the same
sampled event trace. This common-random-number design compares their decisions
under the same demand realization. A separate paired-pulse condition appends
eight rental attempts at Prince-Arthur at 07:47 and eight return attempts at de
Maisonneuve at 07:57 to every trace. These additions are fixed, not extra
Poisson draws, and each succeeds only when inventory or dock space permits. The
feedback rule first uses their inventory consequences at the 08:00 decision.
The frozen controller receives the new observation but ignores it and executes
the action selected before 07:00.

The recorded comparison below uses the same event trace for every controller.
The map, station fills, truck motion, and service failures come from committed
Python trajectories. Controller selection and the playhead only seek through
those records; they do not recompute the plant in the browser.

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

```{include} artifacts/bixi/results.md
```

Feedback reduces service failures in this model, but it does more relocation
work. These are outcomes of the stated simulator, not measurements of service
failures or estimates of a causal effect in BIXI operations. The public archive
omits unsuccessful attempts, historical inventories, and operator truck
movements. The selected morning was chosen after inspection because its
imbalance is easy to see; it is not an unbiased evaluation sample
{cite}`bixiOpenData2024,montrealBixiTrips,hulot2018bixi`.

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

The BIXI simulator supplies station and truck inventories directly to the
feedback controller. Physical sensors often provide only indirect, noisy
measurements of the state.

## Partial Observability

A transition kernel describes uncertainty in the state itself. What changes
when the controller receives only a noisy or ambiguous function of that state?

The BIXI controller receives the modeled inventories directly. Many controllers
instead receive sensor readings that contain only partial or noisy information
about the predictive state. An observation model relates the hidden state to
the available measurement:

$$
y_t=h(x_t,u_t,\nu_t),
\qquad
\nu_t\sim p_\nu.
$$

Here $p_\nu$ is the distribution assigned to measurement noise. For a linear
Gaussian sensor, $y_t=Cx_t+\nu_t$ with
$\nu_t\sim\mathcal N(0,R)$. A controller based only on $y_t$ may lose information
needed to predict future transitions. Under a known model, the posterior belief

$$
b_t(A)=\Pr(x_t\in A\mid y_{0:t},u_{0:t-1})
$$

assigns a probability to each set $A$ of possible hidden states. This belief is
an **information state**: it retains the predictive content of the observation
history in a form that can be updated after each action and measurement. A
point estimate is often cheaper to use, but it may discard uncertainty that
affects future decisions.

### Camera Stabilization under Partial Observation

Picture a camera on a one-axis gimbal attached to a moving vehicle. The motor
tries to keep the horizon level. A tap can rotate the camera, and a sideways
acceleration of the vehicle can disturb its sensors even when the camera itself
does not rotate. The controller must infer what happened from an inertial
measurement unit rather than from the true angle.

The simulated hidden state is

$$
x=(\theta,\omega,b)
$$

where $\theta$ is the camera angle in radians, $\omega$ is its angular velocity
in radians per second, and $b$ is an offset in the gyroscope reading, also in
radians per second. The camera is balanced about the gimbal axis, so gravity
does not create a rotational torque in this model. Its one-axis torque balance
is

$$
\dot\theta=\omega,
\qquad
J\dot\omega=u-c\omega+\tau_{\mathrm{ext}}(t),
\qquad |u|\leq u_{\max}.
$$

Here $J$ is the camera's moment of inertia about the gimbal axis, measured in
$\mathrm{kg\,m^2}$. A larger $J$ means that the same torque produces less angular
acceleration. The motor torque $u$ and external torque $\tau_{\mathrm{ext}}$ are
measured in $\mathrm{N\,m}$, and $c$ is a viscous rotational-damping coefficient
measured in $\mathrm{N\,m\,s/rad}$. The bound $u_{\max}$ represents the motor's
torque limit.

The bias $b$ belongs to the sensor, not to the mechanical plant. If the camera
is motionless but $b=0.8$ degrees per second, the gyroscope reading remains
centered near $0.8$ degrees per second instead of zero. This persistent offset
is different from the small, rapid fluctuations that vary independently from
one measurement to the next.

Real sensor errors depend on temperature, calibration, and hardware. For this
ten-second simulation, a random walk provides a simple model of an offset that
persists but can drift {cite}`elSheimy2008allan`:

$$
b_{k+1}=b_k+\sigma_b\sqrt{\Delta t}\,\epsilon_k,
\qquad \epsilon_k\sim\mathcal N(0,1).
$$

Conditional on $b_k$, the next bias has mean $b_k$ and its change has standard
deviation $\sigma_b\sqrt{\Delta t}$. Nothing in the model pulls the bias back to
zero. Here $b_0=0.8$ degrees per second and
$\sigma_b=0.02\ (\mathrm{degrees/s})/\sqrt{\mathrm{s}}$, so the standard deviation
of one bias change is only $0.002$ degrees per second at the
$\Delta t=0.01$-second sensor period. The chosen random walk therefore changes
slowly relative to the sample-to-sample measurement noise. It is a reproducible
teaching model, not a claim that every gyroscope drifts in exactly this way.

The gyroscope measures angular velocity with both the persistent bias and
sample-to-sample measurement noise:

$$
y_k^\omega=\omega_k+b_k+\nu_k^\omega.
$$

Integrating these readings turns a rate estimate into an angle estimate. Over
$T$ seconds, an approximately constant bias $b$ contributes roughly $Tb$ to
that estimated angle. A camera that is actually motionless therefore acquires
about eight degrees of estimated rotation after ten seconds when an uncorrected
$0.8$-degree-per-second bias is integrated. Zero-mean measurement noise tends
to fluctuate in both directions; the persistent offset accumulates in one
direction.

The accelerometer creates a different ambiguity. It measures *specific force*,
the force per unit mass sensed by its internal proof mass, rather than reporting
a camera angle directly. When the vehicle is stationary or moves at constant
velocity, the support force opposing gravity supplies a vertical reference. A
sideways vehicle acceleration changes the same sensor reading and can therefore
look like camera tilt {cite}`androidMotionSensors2026`.

The simulation keeps only the two axes in the plane of rotation. Its
accelerometer model is

$$
\begin{bmatrix}y_k^x\\y_k^y\end{bmatrix}
=R(-\theta_k)
\begin{bmatrix}a_{x,k}^{w}\\g\end{bmatrix}+\nu_k^a.
$$

The vector $[a_{x,k}^{w},g]^\top$ contains the vehicle's sideways acceleration
and the gravity reference in world coordinates. The rotation $R(-\theta_k)$
expresses that vector along the camera-mounted sensor axes, and $\nu_k^a$ adds
instantaneous measurement noise. If sideways acceleration and measurement
noise are both zero, the direction of this vector determines $\theta_k$. With
sideways acceleration, the angle inferred from the same two components becomes

$$
\operatorname{atan2}(y_k^x,y_k^y)
=\theta_k+\operatorname{atan2}(a_{x,k}^{w},g).
$$

For example, $a_x^w=3\ \mathrm{m/s^2}$ adds an apparent tilt of about $17$ degrees.
A level camera on a vehicle accelerating sideways can then produce the same
accelerometer direction as a tilted camera on a stationary base. This is the
partial-observation problem: the sensor reading alone does not identify which
hidden situation occurred.

The two sensors fail on different time scales. The accelerometer gives a stable
long-run reference when vehicle translation is mild, but a lateral acceleration
corrupts it immediately. The gyroscope tracks rapid rotations without confusing
them with translation, but integrating an unknown bias produces long-run drift.
A **complementary observer** is a recursive state estimator that combines these
two time scales. It uses the gyroscope for rapid changes, the accelerometer for
slow angle correction, and the persistent disagreement between them to estimate
gyroscope bias {cite}`mahony2008complementary`.

The comparison below changes only the estimator. One controller uses the
accelerometer direction as its angle estimate, one integrates the raw gyroscope,
and one uses the complementary observer. Every run receives the same mechanical
plant, noise realization, disturbances, torque limit, and feedback law:

$$
u_k=\operatorname{clip}
\left(-0.9\hat\theta_k-0.12\hat\omega_k,
-0.18,0.18\right).
$$

The function $\operatorname{clip}$ limits the requested motor torque to the
interval $[-0.18,0.18]\ \mathrm{N\,m}$. The simulated plant uses
$J=0.018\ \mathrm{kg\,m^2}$, $c=0.025\ \mathrm{N\,m\,s/rad}$,
one-millisecond mechanical integration, and ten-millisecond sensing and
control. The ten-second sensor trace is generated once with seed 11 and then
supplied unchanged to every estimator.

All three runs start with the camera tilted by eight degrees. A
$0.14\ \mathrm{N\,m}$ tap from 1.50 to 1.62 seconds physically rotates the camera,
so both inertial sensors should help the controller respond. From 4.0 to 4.6
seconds, the base instead accelerates sideways at up to $3\ \mathrm{m/s^2}$. That
event changes the accelerometer reading without directly rotating the balanced
camera. The final quiet interval exposes the angle error produced by integrating
the biased gyroscope.

The replay displays curves only up to the current playhead. Its metrics use the
hidden simulated angle, which is available to the experiment but not to the
controllers.

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

The results follow the two failure mechanisms. The accelerometer-only estimate
reacts strongly to the translation pulse. The integrated-gyroscope estimate
largely ignores that pulse but drifts during the quiet interval. The
complementary observer reduces both errors by using the accelerometer as a slow
correction to the gyroscope-based angle. Sustained unknown translation can still
be mistaken for tilt. These are results from a transparent teaching simulation,
not measured performance of a commercial gimbal.

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

State, action, disturbance, and observation equations describe the model's
semantics. An algorithm still needs a computational way to use that model. The
next distinction concerns which operations the model exposes: formulas and
derivatives, reset-and-step simulation, fixed logged transitions, or new online
interaction.

## Summary and Outlook

Transition kernels separate the action selected by the decision maker from the
disturbance that selects a realized successor. Observation models add a second
separation: the state supports prediction, while the measurement supplies only
the information available for choosing the next action.

These equations specify what a model means, but they do not specify how an
algorithm may use it. Can the algorithm differentiate the transition, reset a
simulator, resample a state, or only inspect logged transitions? [Model
interfaces and learned components](model-interfaces.md) make those operations
explicit.

## Exercises

:::{exercise} Conservation and feedback in bicycle relocation
:label: ex-dynamics-bixi-conservation

Before customer events occur, verify that relocation preserves the sum of
station and truck inventories. Then compare the frozen schedule with inventory
feedback after the declared demand pulse. Which new observation can change the
feedback action, and why can the frozen action not change?
:::

:::{solution} ex-dynamics-bixi-conservation
:class: dropdown

Since $q_{i,0}=s_{i,k}+\rho_{i,k}$ and
$b_{k+1}=b_k-\sum_i\rho_{i,k}$,

$$
\sum_i q_{i,0}+b_{k+1}=\sum_i s_{i,k}+b_k.
$$

Customer trips may later cross the three-station boundary and change that
total. At 08:00, feedback observes the inventories produced by the unexpected
Prince-Arthur rentals and de Maisonneuve returns, so it can alter its transfer
or route. The frozen schedule was fixed from information available before
07:00 and has no rule for using the new observation.
:::

:::{exercise} Tilt or translation?
:label: ex-dynamics-gimbal-ambiguity

Construct two pairs $(\theta,a_x^w)$ that produce the same noiseless
accelerometer-derived tilt. What additional information does the gyroscope
supply, and which ambiguity remains?
:::

:::{solution} ex-dynamics-gimbal-ambiguity
:class: dropdown

Both

$$
(\theta,a_x^w)=(10^\circ,0)
\quad\text{and}\quad
(\theta,a_x^w)=\left(0,g\tan 10^\circ\right)
$$

produce an apparent tilt of $10^\circ$. The gyroscope measures angular rate
plus bias, which helps distinguish a rapid rotation from a translational pulse.
An unknown gyro bias still accumulates as angle error, and sustained unknown
translation remains confounded with gravity in the accelerometer.
:::

:::{exercise} Function-plus-noise and transition kernels
:label: ex-dynamics-reservoir-kernel

Suppose a reservoir with no spill or evaporation obeys
$s_{t+1}=s_t-u_t+w_t$, where
$w_t=\bar w_t+\epsilon_t$. Write a function-plus-noise model and the induced
transition kernel.
:::

:::{solution} ex-dynamics-reservoir-kernel
:class: dropdown

The explicit-noise form is

$$
s_{t+1}=s_t-u_t+\bar w_t+\epsilon_t.
$$

If $\epsilon_t$ has conditional distribution $p_t$, then the induced kernel is

$$
P_t(A\mid s,u)
=\Pr\!\left(s-u+\bar w_t+\epsilon_t\in A\right).
$$

This kernel is the pushforward of the noise distribution through the balance
equation. Adding bounds or spill would change the map and could create
probability mass at a boundary.
:::

:::{exercise} A non-Markov observation
:label: ex-dynamics-nonmarkov-observation

An inference controller observes queue counts and total cache occupancy but not
the remaining output length of each active request. Why are those measurements
not generally a Markov state? What additional state or history would improve
prediction?
:::

:::{solution} ex-dynamics-nonmarkov-observation
:class: dropdown

Two request sets can have equal queue counts and cache occupancy but different
remaining output lengths. Their completion times and cache releases therefore
have different conditional distributions. A full state can retain every
request's remaining work. When that work is hidden, the complete observation
history is sufficient in principle, and a posterior distribution over the
request-level state is a compact information state under a known model.
:::

:::{exercise} Dispatch under hidden service times
:label: ex-dynamics-modeling-dispatch

A computing service reviews its operation once per minute. Four workers can
process jobs from either of two queues. At each review, the dispatcher sees the
number of waiting jobs in each queue, the number of busy workers, the current
worker modes, and a power measurement. It chooses how many workers will be
active and which queue receives priority during the next minute. New jobs
arrive at random times. Each busy job has an unobserved remaining processing
time, and this quantity affects whether the job finishes during the next
minute. The objective penalizes waiting time and energy use. A hard power limit
must hold at every step.

(a) Draw the system boundary and classify the state, observation, action,
disturbance, objective, constraint, time representation, and horizon. Use a
finite horizon of 60 decisions.

(b) Explain why the observed queue counts and number of busy workers are not
generally a Markov state. Give one full-state description for the system and
one information state available to the dispatcher.

(c) Choose between deterministic dynamics and a stochastic transition kernel.
State which sentence in the description determines your choice.

(d) Distinguish a fixed 60-minute staffing schedule from a feedback policy.
Which observations can affect the feedback action after the first minute?

(e) Suppose the only available data are transitions logged under one fixed
staffing schedule. Which model interface does this provide, and what can it not
answer directly about a new feedback policy?
:::

:::{solution} ex-dynamics-modeling-dispatch
:class: dropdown

(a) The boundary contains the two queues, four workers, and their power-relevant
operation. A full state contains the queued jobs, every active job's worker and
remaining processing time, the worker modes, and any variable needed to predict
power. The stated observation contains the two queue counts, busy-worker count,
worker modes, and measured power. The action is the active-worker count and
queue priority. Arrivals and unobserved job requirements are disturbances. The
stage cost combines waiting and energy, and the power cap is a hard constraint.
Time is discrete in one-minute steps over 60 decisions.

(b) Equal counts can hide different remaining workloads and therefore different
completion distributions. The full state just described is Markov under the
declared job model. The complete observation-action history is available to the
dispatcher; under a known stochastic model, its posterior over hidden job
states is a more compact information state.

(c) A stochastic transition kernel is the natural representation because jobs
arrive at random times and hidden processing requirements make completions
uncertain. A deterministic function remains possible only after the relevant
noise realizations are supplied as additional inputs.

(d) The fixed schedule selects all 60 actions from initial information. A
feedback policy may change the active-worker count or priority after observing
new queue counts, busy-worker status, modes, or power.

(e) Those data supply a logged-transition interface under one behavior
schedule. It cannot directly answer how actions absent from that log would
change queues or power, so evaluating a new feedback policy requires coverage
and identification assumptions or a separate transition model.
:::

:::{exercise} Practice under latent mastery
:label: ex-dynamics-modeling-practice

An online practice system presents one problem at each of 20 rounds. Before
round $t$, a learner's unobserved mastery $M_t$ is one of three ordered
categories. The system chooses an easy, medium, or hard problem $A_t$, then
observes correctness and response time $Y_t$. The observation distribution
depends only on $M_t$ and $A_t$. Mastery then changes randomly according to a
distribution that depends only on $M_t$ and $A_t$, producing $M_{t+1}$. The
objective is to maximize the probability that $M_{20}$, after the twentieth
transition, is the highest category, subject to presenting at most five hard
problems. A resettable simulator can sample a transition and observation from
any supplied mastery level and difficulty, but it exposes no equations or
derivatives.

(a) Classify the latent state, observation, action, stochastic dynamics,
objective, constraint, horizon, and model interface. Include any bookkeeping
variable needed to enforce the five-hard-problem limit.

(b) Compare three possible inputs to the decision rule: the most recent answer,
the complete observation history, and the posterior distribution over mastery
levels. Which is an information state under the stated conditional
relationships? Explain why.

(c) Write the structural form of the state transition, observation model, and
feedback policy without inventing numerical probabilities.

(d) Describe an open-loop problem sequence and a feedback problem sequence.
Which part of the available information can change the latter?

(e) Name one computation that the resettable simulator supports directly and
one operation that would require an additional interface.
:::

:::{solution} ex-dynamics-modeling-practice
:class: dropdown

(a) The augmented latent state is $(M_t,c_t,t)$, where $c_t$ counts hard
problems already used. The observation is correctness and response time, and
the action is difficulty. The model has a stochastic mastery transition and a
stochastic observation map. The terminal objective is
$\Pr(M_{20}=\text{highest})$, with $c_t\leq5$ as a hard constraint. The horizon
contains 20 actions. The available model is a generative reset-and-sample
simulator.

(b) The latest answer alone generally loses information from earlier rounds.
The complete history is sufficient but grows with time. Under the stated
conditional relationships, the posterior
$b_t(m)=\Pr(M_t=m\mid Y_{0:t-1},A_{0:t-1})$, augmented by $c_t$ and $t$, is a
Markov information state.

(c) Without assigning numbers, the structure is

$$
Y_t\sim O(\,\cdot\mid M_t,A_t),\qquad
M_{t+1}\sim P(\,\cdot\mid M_t,A_t),\qquad
c_{t+1}=c_t+\mathbf 1\{A_t=\mathrm{hard}\},
$$

with a feasible feedback policy
$A_t=\pi_t(b_t,c_t)$ that excludes the hard action once $c_t=5$.

(d) An open-loop sequence fixes all 20 difficulties before any answers are
seen. A feedback sequence selects the next difficulty after updating the
mastery posterior from correctness and response time, while retaining the
remaining hard-problem budget.

(e) The simulator directly supports arbitrary one-step samples and Monte Carlo
rollouts from chosen latent states and actions. Exact probabilities, analytic
derivatives, or symbolic local equations require an additional interface.
:::
