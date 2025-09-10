## Dynamics Models for Decision Making

The kind of model we need here is a **dynamics model**. It does not just describe correlations. It tells us how a system **evolves in time** and, most importantly for control, how that evolution **responds to inputs** we choose.

A dynamics model earns its keep by answering counterfactuals of the form: *given an initial condition and an input schedule, what trajectory should I expect?* That ability to roll a trajectory forward under different candidate inputs is the backbone of planning, policy evaluation, and learning from interaction.

At this level, we can think of the model as a trajectory generator:

$$
(\mathbf{x}_0,\ \{\mathbf{u}_t\},\ \{\mathbf{d}_t\}) \ \longmapsto\ \{\mathbf{x}_t,\ \mathbf{y}_t\}_{t=0:T},
$$

where $\mathbf{u}_t$ are **controls** we set, $\mathbf{d}_t$ are **exogenous drivers** we do not control (weather, inflow, demand), $\mathbf{x}_t$ are internal **system variables**, and $\mathbf{y}_t$ are **observations**. The split between $\mathbf{u}$ and $\mathbf{d}$ is practical: it separates what we can act on from what we must accommodate.

Two design pressures shape such models:

1. **Responsiveness to inputs.** The model must expose the levers that matter for the decision problem, even if everything underneath is approximate.
2. **Memory management.** To simulate step by step, we need a compact summary of the past that is sufficient to predict the next step once an input arrives. That summary is what we will call the **state**.

This brings us to a standard but powerful representation. Rather than carry the full history, we look for a variable $\mathbf{x}_t$ that captures "what matters so far" for predicting what comes next under a given input. With that variable in hand, the model advances in small increments and can be composed with estimators and controllers.

With this motivation in place, we can now introduce the formalism.

## The State‑Space Perspective

Most dynamics models, whether derived from physics or learned from data, can be cast into **state‑space form**. The state $\mathbf{x}$ is the compact memory that summarizes the past for prediction and control. Inputs $\mathbf{u}$ perturb that state, exogenous drivers $\mathbf{d}$ push it around, and outputs $\mathbf{y}$ are what we can measure. The equations look the same whether time is treated in discrete steps or as a continuous variable.

### Discrete versus continuous time

How we represent time is dictated by how we sense and actuate: digital controllers sample and apply inputs in steps; the underlying physics evolve continuously.

Time can be represented in two complementary ways, depending on how the system is sensed, actuated, or modelled.

In **discrete time**, we treat time as an integer counter, $t = 0, 1, 2, \dots$, advancing in fixed steps. This matches how digital systems operate: sensors are sampled periodically, decisions are made at regular intervals, and most logged data takes this form.

**Continuous time** treats time as a real variable, $t \in \mathbb{R}_{\ge 0}$. Many physical systems (mechanical, thermal, chemical) are most naturally expressed this way, using differential equations to describe how state changes.

The two views are interchangeable to some extent. A continuous-time model can be discretized through numerical integration, although this involves approximation. The degree of approximation depends on both the step size $\Delta t$ and the integration algorithm used. Conversely, a discrete-time policy can be extended to continuous time by holding inputs constant over time intervals (a zeroth-order hold), or by interpolating between values.

In physical systems, this hybrid setup is almost always present. Control software sends discrete commands to hardware (say, the output of a PID controller) which are then processed by a DAC (digital-to-analog converter) and applied to the plant through analog signals. The hardware might hold a voltage constant, ramp it, or apply some analog shaping. On the sensing side, continuous signals are sampled via ADCs before reaching a digital controller. So in practice, even systems governed by continuous dynamics end up interfacing with the digital world through discrete-time approximations.

This raises a natural question: if everything eventually gets discretized anyway, why not just model everything in discrete time from the start?

In many cases, we do. But continuous-time models can still be useful, sometimes even necessary. They often make physical assumptions more explicit, connect more naturally to domain knowledge (e.g. differential equations in mechanics or thermodynamics), and expose invariances or conserved quantities that get obscured by time discretization. They also make it easier to model systems at different time scales, or to reason about how behaviors change as resolution increases. So while implementation happens in discrete time, thinking in continuous time can clarify the structure of the model.

Still, it's helpful to see how both representations look in mathematical form. The state-space equations are nearly identical with different notations depending on how time is represented.

**Discrete time**

Having defined state as the summary we carry forward, a step of prediction applies the chosen input and advances the state.

$\mathbf{x}_{t+1} = f_t(\mathbf{x}_t, \mathbf{u}_t), \qquad \mathbf{y}_t = h_t(\mathbf{x}_t, \mathbf{u}_t).$

**Continuous time**

$\dot{\mathbf{x}}(t) = f(\mathbf{x}(t), \mathbf{u}(t)), \qquad \mathbf{y}(t) = h(\mathbf{x}(t), \mathbf{u}(t)).$

The dot denotes a derivative with respect to real time; everything else (state, control, observation) remains the same.

When the functions $f$ and $h$ are linear we obtain

Linearity is not a belief about the world, it is a modeling choice that trades fidelity for transparency and speed.

$\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u}, \qquad \mathbf{y} = C\mathbf{x} + D\mathbf{u}.$

The matrices $A, B, C, D$ may vary with $t$.  Readers with an ML background will recognise the parallel with recurrent neural networks: the state is the hidden vector, the control the input, and the output the read‑out layer.

Classical control often moves to the frequency domain, using Laplace and Z‑transforms to turn differential and difference equations into algebraic ones. That is invaluable for stability analysis of linear time‑invariant systems, but the time‑domain state‑space view is more flexible for learning and simulation, so we will keep our primary focus there.


## Examples of Deterministic Dynamics: HVAC Control

Imagine you're in Montréal, in the middle of February. Outside it's -20°C, but inside your home, a thermostat tries to keep things comfortable. When the indoor temperature drops below your setpoint, the heating system kicks in. That system (a small building, a heater, the surrounding weather) can be modeled mathematically.

We start with a very simple approximation: treat the entire room as a single "thermal mass," like a big air-filled box that heats up or cools down depending on how much heat flows in or out.

Let $\mathbf{x}(t)$ be the indoor air temperature at time $t$, and $\mathbf{u}(t)$ be the heating power supplied by the HVAC system. The outside air temperature, denoted $\mathbf{d}(t)$, affects the system too, acting as a known disturbance. Then the rate of change of indoor temperature is:

$$
\dot{\mathbf{x}}(t) = -\frac{1}{RC}\mathbf{x}(t) + \frac{1}{RC}\mathbf{d}(t) + \frac{1}{C}\mathbf{u}(t).
$$

Here:

* $R$ is a thermal resistance: how well the walls insulate.
* $C$ is a thermal capacitance: how much energy it takes to heat the air.

This is a **continuous-time linear system**, and we can write it in standard state-space form:

$$
\dot{\mathbf{x}}(t) = \mathbf{A}\mathbf{x}(t) + \mathbf{B}\mathbf{u}(t) + \mathbf{E}\mathbf{d}(t), \quad \mathbf{y}(t) = \mathbf{C}\mathbf{x}(t),
$$

with:

* $\mathbf{x}(t)$: indoor air temperature (the state)
* $\mathbf{u}(t)$: heater input (the control)
* $\mathbf{d}(t)$: outdoor temperature (disturbance)
* $\mathbf{y}(t)$: observed indoor temperature (output)
* $\mathbf{A} = -\frac{1}{RC}$
* $\mathbf{B} = \frac{1}{C}$
* $\mathbf{E} = \frac{1}{RC}$
* $\mathbf{C} = 1$

This model is simple, but too simplistic. It ignores the fact that the walls themselves store heat and release it slowly. This kind of delay is called **thermal inertia**: even if you turn the heater off, the walls might continue to warm the room for a while.

To capture this effect, we need to expand our state to include the wall temperature. We now model two coupled thermal masses: one for the air, and one for the wall. Heat can flow from the heater into the air, from the air into the wall, and from the wall out to the environment. This gives a more realistic description of how heat moves through a building envelope.

We write down an energy balance for each mass:

* For the air:

$$
C_{\text{air}} \frac{dT_{\text{in}}}{dt} = \frac{T_{\text{wall}} - T_{\text{in}}}{R_{\text{ia}}} + u(t),
$$

* For the wall:

$$
C_{\text{wall}} \frac{dT_{\text{wall}}}{dt} = \frac{T_{\text{out}} - T_{\text{wall}}}{R_{\text{wo}}} - \frac{T_{\text{wall}} - T_{\text{in}}}{R_{\text{ia}}}.
$$

Each term on the right-hand side corresponds to a flow of heat: the air gains heat from the wall and the heater, and the wall exchanges heat with both the air and the outside.

Now define the state vector:

$$
\mathbf{x}(t) = \begin{bmatrix} T_{\text{in}}(t) \\ T_{\text{wall}}(t) \end{bmatrix},
\quad \mathbf{u}(t) = u(t),
\quad \mathbf{d}(t) = T_{\text{out}}(t).
$$

Dividing both equations by their respective capacitances and rearranging terms, we arrive at the coupled system:

$$
\dot{\mathbf{x}}(t) = \mathbf{A}\mathbf{x}(t) + \mathbf{B}\mathbf{u}(t) + \mathbf{E}\mathbf{d}(t), \quad \mathbf{y}(t) = \mathbf{C}\mathbf{x}(t),
$$

with:

$$
\mathbf{A} = \begin{bmatrix}
-\frac{1}{R_{\text{ia}}C_{\text{air}}} & \frac{1}{R_{\text{ia}}C_{\text{air}}} \\
\frac{1}{R_{\text{ia}}C_{\text{wall}}} & -\left(\frac{1}{R_{\text{ia}}} + \frac{1}{R_{\text{wo}}}\right) \frac{1}{C_{\text{wall}}}
\end{bmatrix},
\quad
\mathbf{B} = \begin{bmatrix} \frac{1}{C_{\text{air}}} \\ 0 \end{bmatrix},
\quad
\mathbf{E} = \begin{bmatrix} 0 \\ \frac{1}{R_{\text{wo}}C_{\text{wall}}} \end{bmatrix},
\quad
\mathbf{C} = \begin{bmatrix} 1 & 0 \end{bmatrix}.
$$

Each entry in $\mathbf{A}$ has a physical interpretation:

* $A_{11}$: heat loss from the air to the wall
* $A_{12}$: heat gain by the air from the wall
* $A_{21}$: heat gain by the wall from the air
* $A_{22}$: net loss from the wall to both the air and the outside

The temperatures are now dynamically coupled: any change in one affects the other. The wall acts as a buffer that absorbs and releases heat over time.

This is still a linear system, just with a 2D state. But already it behaves differently. The walls absorb and release heat, smoothing out fluctuations and slowing down the system's response.

As we add more rooms, walls, or building elements, the system grows. Each new temperature adds a new state. The equations still have the same structure, and their sparsity follows the building layout. Nodes represent temperatures; edges encode how heat flows between them.

### What Do We Control?
This network of states is what we control. What we mean by “control input” $\mathbf{u}(t)$ depends on both what we want to achieve and what we can implement in practice.

The most direct interpretation is to let $\mathbf{u}(t)$ represent the actual heating power delivered to the system, measured in watts. This makes sense when modeling from physical principles or simulating a system with fine-grained actuation.

In many real buildings, however, thermostats don’t issue power commands. They activate a relay, turning the heater on or off based on whether the measured temperature crosses a setpoint. Some systems allow for modulated control—such as varying fan speed or partially opening a valve—but those details are often hidden behind firmware or closed controllers.

A common implementation involves a **PID control loop** that compares the measured temperature to a setpoint and adjusts the control signal accordingly. While the actual logic might be simple, the resulting behavior appears smoothed or delayed from the perspective of the building.

Depending on the abstraction level, we might:

* Treat $\mathbf{u}(t)$ as continuous power input, if designing the full control logic.
* Use it as a setpoint input, assuming a lower-level controller handles the rest.
* Or reduce it to a binary signal—heater on or off—when working with logged behavior from a smart thermostat.

Each perspective shapes the kind of model we build and the kind of control problem we pose. If we're aiming to design a controller from scratch, it may be worth modeling the full closed-loop dynamics. If the goal is to tune setpoints or learn policies from data, a coarser abstraction might be not only sufficient, but more robust.

### Why This Model?

At this point, you might wonder: why go through the trouble of building this kind of physics-based model at all? After all, if we can log indoor temperatures, thermostat actions, and weather data, isn’t it easier to just learn a model from data? A neural ODE, for example, would let us define a parameterized function:

$$
\dot{\mathbf{z}}(t) = f_{\boldsymbol{\theta}}(\mathbf{z}(t), \mathbf{u}(t), \mathbf{d}(t)), \quad \mathbf{y}(t) = g_{\boldsymbol{\theta}}(\mathbf{z}(t)),
$$

with both $f_{\boldsymbol{\theta}}$ and $g_{\boldsymbol{\theta}}$ learned from data. The internal state $\mathbf{z}(t)$ is not tied to any physical quantity. It just needs to be expressive enough to explain the observations.

That flexibility can be useful, particularly when a large dataset is already available. But in building control and energy modeling, the constraints are usually different.

Often, the engineer or consultant on site is working under tight time and information budgets. A floor plan might be available, along with some basic specs on insulation or window types, and a few days of logged sensor data. The task might be to simulate load under different weather scenarios, tune a controller, or just help understand why a room is slow to heat up. The model has to be built quickly, adapted easily, and remain understandable to others working on the same system.

In that context, RC models are often the default choice: not because they are inherently better, but because they fit the workflow.

**Interpretability.**
The parameters correspond to things you can reason about: thermal resistance, capacitance, heat transfer between zones. You can cross-check values against architectural plans, or adjust them manually when something doesn’t line up. You can tell which wall or zone is contributing to slow recovery times.

**Identifiability with limited data.**
RC models can often be calibrated from short data traces, even when not all state variables are directly observable. The structure already imposes constraints: heat flows from hot to cold, dynamics are passive, responses are smooth. Those properties help narrow the space of valid parameter settings. A neural ODE, in contrast, typically needs more data to settle into stable and plausible dynamics—especially if no additional constraints are enforced during training.

**Simplicity and reuse.**
Once the model is built, it’s straightforward to modify. If a window is replaced, or a wall gets insulated, you only need to update a few numbers.  It’s easy to pass along to another engineer or embed in a larger simulation. A model like

$$
\dot{\mathbf{x}} = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u} + \mathbf{E}\mathbf{d}
$$
is linear and low-dimensional. Simulating it is cheap, even if you do it many times. That may not matter now, but it will matter later, when we want to optimize over trajectories or learn from them.

This doesn’t mean RC models are always sufficient. They simplify or ignore many effects: solar gains, occupancy, nonlinearities, humidity, equipment switching behavior. If those effects are significant, and you have enough data, a black-box model (neural ODE or otherwise) might achieve lower prediction error. In practice, though, it’s common to combine the two: use the RC structure as a backbone, and learn a residual model to correct for unmodeled dynamics.

## From Deterministic to Stochastic

The models we've seen so far were deterministic: given an initial state and input sequence, the system evolves in a fixed, predictable way. But real systems rarely behave so neatly. Sensors are noisy. Parameters drift. The world changes in ways we can't fully model.

To account for this uncertainty, we move from deterministic dynamics to **stochastic models**. There are two equivalent but conceptually distinct ways to do this.

### Function plus Noise

The most direct extension adds a noise term to the dynamics:

$$
\mathbf{x}_{t+1} = f_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t), \quad \mathbf{w}_t \sim p_{\mathbf{w}}.
$$

If the noise is additive and Gaussian, we recover the standard linear-Gaussian setup used in Kalman filtering:

$$
\mathbf{x}_{t+1} = A\mathbf{x}_t + B\mathbf{u}_t + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(0, Q).
$$

But we’re not restricted to Gaussian or additive noise. For instance, if the noise distribution is non-Gaussian:

$$
\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t) + \mathbf{w}_t, \quad \mathbf{w}_t \sim \text{Laplace}, \ \text{or}\ \text{Student-t},
$$

then $\mathbf{x}_{t+1}$ inherits those properties. This is known as a **convolution model**: the next-state distribution is a shifted version of the noise distribution, centered around the deterministic prediction. More formally, it's a special case of a **pushforward measure**: the randomness from $\mathbf{w}_t$ is “pushed forward” through the function $f$ to yield a distribution over outcomes. 

Or the noise might enter multiplicatively:

$$
\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t) + \Gamma(\mathbf{x}_t, \mathbf{u}_t) \mathbf{w}_t,
$$

where $\Gamma$ is a matrix that modulates the effect of the noise, potentially depending on state and control. If $\Gamma$ is invertible, we can even write down an explicit density via a change-of-variables:

$$
p(\mathbf{x}_{t+1} \mid \mathbf{x}_t, \mathbf{u}_t) = p_{\mathbf{w}}\left(\Gamma^{-1}(\mathbf{x}_t, \mathbf{u}_t)\left[\mathbf{x}_{t+1} - f(\mathbf{x}_t, \mathbf{u}_t)\right] \right)\cdot \left| \det \Gamma^{-1} \right|.
$$

This kind of structured noise is common in practice, for example, when disturbances are amplified at certain operating points.

The **function-plus-noise** view is natural when we have a physical or simulator-based model and want to account for uncertainty around it. It is **constructive**: we know how the system evolves and how the randomness enters. This means we can **track the source of variability along a trajectory**, which is particularly useful for techniques like **reparameterization** or **infinitesimal perturbation analysis (IPA)**. These methods rely on being able to differentiate through the noise injection mechanism, something that is much easier when the noise is explicit and structured. 

### Transition Kernel

The second perspective skips over the internal noise and defines the system directly in terms of the probability distribution over next states:

$$
p(\mathbf{x}_{t+1} \mid \mathbf{x}_t, \mathbf{u}_t).
$$

This **transition kernel** encodes all the uncertainty in the system's evolution, without reference to any underlying noise source or functional form.

This view is strictly more general: it includes the function-plus-noise case as a special instance. If we do know the function $f$ and the noise distribution $p_{\mathbf{w}}$ from the generative model, then the transition kernel is obtained by “pushing” the randomness through the function:

$$
p(\mathbf{x}_{t+1} \mid \mathbf{x}_t, \mathbf{u}_t) = \int \delta(\mathbf{x}_{t+1} - f(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w})) \, p_{\mathbf{w}}(\mathbf{w}) \, d\mathbf{w}.
$$

This might look abstract, but it’s just marginalization: for each possible noise value $\mathbf{w}$, we compute the resulting next state, and then average over all possible $\mathbf{w}$, weighted by how likely each one is.

If the noise were discrete, this becomes a sum:

$$
p(\mathbf{x}_{t+1} \mid \mathbf{x}_t, \mathbf{u}_t) = \sum_{i=1}^k \mathbb{1}\{f(\mathbf{x}_t, \mathbf{u}_t, w_i) = \mathbf{x}_{t+1}\} \cdot p_i
$$

This abstraction is especially useful when we don’t know (or don’t care about) the underlying function or noise distribution. All we need is the ability to sample transitions or estimate their likelihoods. This is the default formulation in reinforcement learning, econometrics, and other settings focused on behavior rather than mechanism.

### Continuous-Time Analogue

In continuous time, the stochastic dynamics of a system are often described using a **stochastic differential equation (SDE)**:

$$
d\mathbf{X}_t = f(\mathbf{X}_t, \mathbf{U}_t)\,dt + \sigma(\mathbf{X}_t, \mathbf{U}_t)\,d\mathbf{W}_t,
$$

where $\mathbf{W}_t$ is Brownian motion. The first term, called the **drift**, describes the average motion of the system. The second, scaled by $\sigma$, models how random fluctuations (diffusion) enter over time. Just like in discrete time, this is a **function + noise** model: the state evolves through a deterministic path perturbed by stochastic input.

This generative view again induces a probability distribution over future states. At any future time $t + \Delta t$, the system doesn’t land at a single state but is described by a distribution that depends on the initial condition and the noise along the way.

Mathematically, this distribution evolves according to what's called the **Fokker–Planck equation**—a partial differential equation that governs how probability density “flows” through time. It plays the same role here as the transition kernel did in discrete time: describing how likely the system is to be in any given state, without referring to the noise directly.

While the mathematical generalization is clean, working with continuous-time stochastic models can be more challenging. Simulating sample paths is often straightforward (eg. nowadays diffusion models in generative AI), but writing down or computing the exact transition distribution usually isn’t. That’s why many practical methods still rely on discrete-time approximations, even when the underlying system is continuous.

#### Example: Managing a Québec Hydroelectric Reservoir

On the James Bay plateau, 1 400 km north of Montréal, the Robert-Bourassa reservoir stores roughly 62 km³ of water, more than the volume of Lake Ontario above its minimum operating level. Sixteen giant turbines sit 140 m below the surface, converting that stored head into 5.6 GW of electricity, about a fifth of Hydro-Québec’s total capacity. A steady share of that output feeds Québec’s aluminium smelters, which depend on stable, uninterrupted power.

Water managers face competing objectives:

* **Flood safety.** Sudden snowmelt or storms can overfill the basin, forcing emergency spillways to open. These events are spectacular, but carry real downstream risk and economic cost.
* **Energy reliability.** If the level falls too low, turbines sit idle and contracts go unmet. Voltage dips at the smelters are measured in lost millions.

A basic deterministic model for the reservoir’s mass balance is just bookkeeping:

$$
\mathbf{x}_{t+1} = \mathbf{x}_t + \mathbf{r}_t - \mathbf{u}_t,
$$

where $\mathbf{x}_t$ is the current reservoir level, $\mathbf{u}_t$ is the controlled outflow through turbines, and $\mathbf{r}_t$ is the natural inflow from rainfall and upstream runoff.

But inflow is variable, and its statistical structure matters. Two hydrological regimes dominate:

* In spring, melting snow over days can produce a long-tailed inflow distribution, often modeled as log-normal or Gamma.
* In summer, convective storms yield a skewed mixture: a point mass at zero (no rain), and a thin but heavy tail capturing sudden bursts.

This motivates a simple stochastic extension:

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \mathbf{u}_t + \mathbf{w}_t, \quad
\mathbf{w}_t \sim
\begin{cases}
0 & \text{with prob. } p_0, \\\\
\text{LogNormal}(\mu, \sigma^2) & \text{with prob. } 1 - p_0.
\end{cases}
$$

Here the physics is fixed, and all uncertainty sits in the inflow term $\mathbf{w}_t$. Rather than fitting a full transition model from $(\mathbf{x}_t, \mathbf{u}_t)$ to $\mathbf{x}_{t+1}$, we can isolate the inflow by rearranging the mass balance:

$$
\hat{\mathbf{w}}_t = \mathbf{x}_{t+1} - \mathbf{x}_t + \mathbf{u}_t.
$$

This gives a direct estimate of the realized inflow at each timestep. From there, the problem becomes one of density estimation: fit a probabilistic model to the residuals $\hat{\mathbf{w}}_t$. In spring, this might be a log-normal distribution. In summer, a two-part mixture: a point mass at zero, and an exponential tail. These distributions can be estimated by maximum likelihood, or adjusted using additional features (covariates) such as upstream snowpack or forecasted temperature.

This setup has practical benefits. Fixing the physical part of the model (how levels respond to inflow and outflow) helps focus the statistical modeling effort. Rather than fitting a full system model, we only need to estimate the variability in inflows. This reduces the number of degrees of freedom and makes the estimation problem easier to interpret. It also avoids conflating uncertainty in inflow with uncertainty in the system's response.

Compare this to a more generic approach, such as linear regression:

$$
\mathbf{x}_{t+1} = a \mathbf{x}_t + b \mathbf{u}_t + \varepsilon_t.
$$

This is straightforward to fit, but offers no guarantee that the result behaves sensibly. The model might violate conservation of mass, or compensate for inflow variation by adjusting coefficients $a$ and $b$. This can lead to misleading conclusions, especially when extrapolating beyond the training data.

<!-- Hydro‑Québec engineers rely on structured models in practice. Over 150 gauging stations across the La Grande basin report real-time flows, levels, and precipitation to Environment Canada’s HYDAT database, which is accessible through a public API. These data feed into Hydro‑Québec’s SCADA systems, along with snow-course readings and rainfall estimates. From there, engineers build seasonal inflow models and update them daily.

Synthetic years are then generated by sampling from these models. Each sampled inflow sequence is pushed through the deterministic mass balance, producing a possible reservoir trajectory. These Monte Carlo rollouts are used directly for planning. They help evaluate turbine schedules, size safety margins, and identify periods of elevated risk.

Structured models are not just a matter of physical fidelity. They shape how data is used, how uncertainty is handled, and how downstream decisions are informed. The separation between known dynamics and unknown inputs gives a cleaner interface between estimation and control. -->


## Partial Observability

So far, we’ve assumed that the full system state $\mathbf{x}_t$ is available. But in most real-world settings, only a partial or noisy observation is accessible. Sensors have limited coverage, measurements come with noise, and some variables aren’t observable at all.

To model this, we introduce an **observation equation** alongside the system dynamics:

$$
\begin{aligned}
\mathbf{x}_{t+1} &= f_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t), \quad \mathbf{w}_t \sim p_{\mathbf{w}}, \\
\mathbf{y}_t &= h_t(\mathbf{x}_t, \mathbf{v}_t), \quad \mathbf{v}_t \sim p_{\mathbf{v}}.
\end{aligned}
$$

The state $\mathbf{x}_t$ evolves under control inputs $\mathbf{u}_t$ and process noise $\mathbf{w}_t$, but we don’t get to see $\mathbf{x}_t$ directly. Instead, we observe $\mathbf{y}_t$, which depends on $\mathbf{x}_t$ through some possibly nonlinear, noisy function $h_t$. The noise $\mathbf{v}_t$ captures measurement uncertainty.

This setup defines a partially observed system. Even if the underlying dynamics are known, we still face uncertainty due to limited visibility into the true state. The controller or estimator must rely on the observations $\mathbf{y}_{0\:t}$ to make sense of the hidden trajectory.

In the **deterministic** case, if the output map $h_t$ is full-rank and invertible, we may be able to reconstruct the state directly from the output: no filtering required. But once noise is introduced, that invertibility becomes more subtle: even if $h_t$ is bijective, the presence of $\mathbf{v}_t$ prevents us from recovering $\mathbf{x}_t$ exactly. In this case, we must shift from inversion to estimation, often via probabilistic inference.

In the **linear-Gaussian case**, the model simplifies to:

$$
\begin{aligned}
\mathbf{x}_{t+1} &= A\mathbf{x}_t + B\mathbf{u}_t + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(0, Q), \\
\mathbf{y}_t &= C\mathbf{x}_t + D\mathbf{u}_t + \mathbf{v}_t, \quad \mathbf{v}_t \sim \mathcal{N}(0, R).
\end{aligned}
$$

This is the classical state-space model used in signal processing and control. It’s fully specified by the system matrices and the covariances $Q$ and $R$. The state is no longer known, but under these assumptions it can be estimated recursively using the **Kalman filter**, which maintains a Gaussian belief over $\mathbf{x}_t$.

Even when the model is nonlinear or non-Gaussian, the structure remains the same: a dynamic state evolves, and a separate observation process links it to the data we see. Many modern estimation techniques, including extended and unscented Kalman filters, particle filters, and learned neural estimators, build on this core structure.

### Observation Kernel View

Just as we moved from function-based dynamics to transition kernels, we can abstract away the noise source and define the **observation distribution** directly:

$$
p(\mathbf{y}_t \mid \mathbf{x}_t).
$$

This kernel summarizes what the sensors tell us about the hidden state. If we know the generative model—say, that $\mathbf{y}_t = h_t(\mathbf{x}_t) + \mathbf{v}_t$ with known $p_{\mathbf{v}}$—then this kernel is induced by marginalizing out $\mathbf{v}_t$:

$$
p(\mathbf{y}_t \mid \mathbf{x}_t) = \int \delta\bigl(\mathbf{y}_t - h_t(\mathbf{x}_t, \mathbf{v})\bigr)\, p_{\mathbf{v}}(\mathbf{v})\, d\mathbf{v}.
$$

But we don’t have to start from the generative form. In practice, we might define or learn $p(\mathbf{y}_t \mid \mathbf{x}_t)$ directly, especially when dealing with black-box sensors, perception models, or abstract measurement processes.

### Example – Stabilizing a Telescope’s Vision with Adaptive Optics

On Earth, even the largest telescopes can’t see perfectly. As starlight travels through the atmosphere, tiny air pockets with different temperatures bend the light in slightly different directions. The result is a distorted image: instead of a sharp point, a star looks like a flickering blob. The distortion happens fast, on the order of milliseconds, and changes continuously as wind moves the turbulent layers overhead.

This is where **adaptive optics (AO)** comes in. AO systems aim to cancel out these distortions in real time. They do this by measuring how the incoming wavefront of light is distorted and using a flexible mirror to apply a counter-distortion that straightens it back out. But there's a catch: you can’t observe the wavefront directly. You only get noisy measurements of its **slopes** (the angles of tilt at various points), and you have to act fast, before the atmosphere changes again.

To design a controller here, we need a model of how the distortions evolve. And that means building a decision-making model: one that includes uncertainty, partial observability, and fast feedback.

**State.** The main object we're trying to track is the distortion of the incoming wavefront. We can’t observe this phase field $\phi(\mathbf{r}, t)$ directly, but we can represent it approximately using a finite basis (e.g., Fourier or Zernike). The coefficients of this expansion form our internal state:

$$
\mathbf{x}_t \in \mathbb{R}^n \quad \text{(wavefront distortion at time } t).
$$

**Dynamics.** The atmosphere evolves in time. A simple but surprisingly effective model assumes the turbulence is "frozen" and just blown across the telescope by the wind. That gives us a **discrete-time linear model**:

$$
\mathbf{x}_{t+1} = \mathbf{A} \mathbf{x}_t + \mathbf{w}_t,
$$

where $\mathbf{A}$ shifts the distortion pattern in space, and $\mathbf{w}_t$ is a small random change from evolving turbulence. This noise is not arbitrary: its statistics follow a power law derived from **Kolmogorov’s turbulence model**. In particular, higher spatial frequencies (small-scale wiggles) have less energy than low ones. That lets us build a prior on how likely different distortions are.

**Observations.** We can’t see the full wavefront. Instead, we use a **wavefront sensor**: a camera that captures how the light bends. What it actually measures are local slopes: the gradients of the wavefront, not the wavefront itself. So our observation model is:

$$
\mathbf{y}_t = \mathbf{C} \mathbf{x}_t + \boldsymbol{\varepsilon}_t,
$$

where $\mathbf{C}$ is a known matrix that maps wavefront distortion to measurable slope angles, and $\boldsymbol{\varepsilon}_t$ is measurement noise (e.g., due to photon limits).

**Control.** Our job is to flatten the wavefront using a deformable mirror. The mirror can apply a small counter-distortion $\mathbf{u}_t$ that subtracts from the atmospheric one:

$$
\text{Residual state:} \quad \mathbf{x}_t^{\text{res}} = \mathbf{x}_t - \mathbf{B} \mathbf{u}_t.
$$

The goal is to choose $\mathbf{u}_t$ to minimize the residual distortion by making the light flat again.

**Why a model matters.** Without a model, we’d just react to the current noisy measurements. But with a model, we can predict how the wavefront will evolve, filter out noise, and act preemptively. This is essential in AO, where decisions must be made every millisecond. Kalman filters are often used to track the hidden state $\mathbf{x}_t$, combining model predictions with noisy measurements, and linear-quadratic regulators (LQR) or other optimal controllers use those estimates to choose the best correction.

**Time structure.** This is a rare case where **continuous-time modeling** also plays a role. The true evolution of the turbulence is continuous, and we can model it using a **stochastic differential equation (SDE)**:

$$
d\mathbf{x}(t) = \mathbf{F} \mathbf{x}(t)\,dt + \mathbf{G}\,d\mathbf{W}(t),
$$

where $\mathbf{W}(t)$ is Brownian motion and the matrix $\mathbf{G}$ encodes the Kolmogorov spectrum. Discretizing this equation gives us the $\mathbf{A}$ and $\mathbf{Q}$ matrices for the discrete-time model above.

<!-- ## Data-Driven Identification

Not all models come from physics. Sometimes, we fit them directly from data.

Even a basic linear regression of the form:

$$
x_{t+1} = a x_t + b u_t + c + \varepsilon_t
$$

is a dynamical model. But things can get more sophisticated. Subspace identification methods, sparse regressions like SINDy, Koopman embeddings, neural ODEs—all of these let us learn models from observed trajectories. The key question is how much structure we assume. Do we enforce linearity? Time-invariance? Do we try to model the noise? -->
<!-- 
# Comparing Physics-Based RC Models with Black-Box Fits

The data used in this experiment comes from the *Building Energy Geeks* repository, an open-source collection created to demonstrate statistical learning techniques in building energy performance. The file `statespace.csv` provides a time series of indoor and outdoor temperatures together with heating power and solar irradiance. While not tied to a specific building description in the repository, it is designed to mimic realistic conditions either from actual sensor measurements or detailed simulation outputs. This dataset serves as a concrete foundation to explore the contrast between physics-based modeling and purely data-driven approaches.

At the core of our study lies the so-called **2R2C model**, a reduced-order representation of building thermal dynamics. The name refers to two resistances and two capacitances arranged in a thermal network that captures how heat flows between the indoor environment, the building envelope, and the outdoors. The indoor air temperature $T_i$ is influenced by the envelope temperature $T_e$, which itself exchanges heat with the external environment at temperature $T_o$. The resistances $R_i$ and $R_o$ describe the ease of heat conduction across these boundaries, while the capacitances $C_i$ and $C_e$ represent the heat storage capacity of the indoor air and of the building mass. By including the effect of heating input $\Phi_H$ and solar gains $\Phi_S$, the model balances both controllable and environmental influences.  

Mathematically, the dynamics are written as a pair of coupled ordinary differential equations. The first governs the indoor air temperature and is given by

$$
\frac{dT_i}{dt} = \frac{T_e - T_i}{R_i C_i} + \frac{\eta_H \Phi_H}{C_i} + \frac{A_i \Phi_S}{C_i},
$$

while the second governs the envelope temperature,

$$
\frac{dT_e}{dt} = \frac{T_i - T_e}{R_i C_e} + \frac{T_o - T_e}{R_o C_e} + \frac{A_e \Phi_S}{C_e}.
$$

Here $\eta_H$ represents the efficiency of the heating system and $A_i, A_e$ are effective areas for solar gains. Since the original dataset is indexed in hours rather than seconds, the right-hand side of both equations must be scaled by a factor of 3600 to ensure correct integration over the chosen time unit.

The task is to identify the parameters of this model from data. To do so, we fit the parameters by minimizing the discrepancy between the simulated indoor temperature $T_i$ and the measured trajectory within a training window of 10 to 40 hours. A robust least-squares method with Huber loss is used so that large deviations, possibly due to noise or outliers, do not dominate the fit. Early time points in the training window are given slightly higher weight to ensure that the transient behavior is captured accurately, which is important when the system is initialized away from equilibrium. Once fitted, the model is simulated forward over the entire 0–100 hour horizon, allowing us to test its predictive power on an unseen window spanning 50 to 90 hours.

To provide meaningful context, we benchmark this physics-based model against two black-box alternatives. The first is a linear regression model that directly maps the contemporaneous values of outdoor temperature, heating power, and solar irradiance to the indoor temperature. This approach ignores temporal dynamics and treats the problem as a purely static regression. The second is a multilayer perceptron (MLP) that is trained with autoregressive lags. Specifically, the MLP is provided with the recent history of indoor temperatures together with the external inputs to predict the next indoor temperature. During training, a technique known as teacher forcing is employed, meaning the true past values of $T_i$ are always supplied, which allows the network to achieve a very tight fit on the training window. However, when rolled out on the test window without access to the ground-truth future values, small prediction errors accumulate, and the model struggles to generalize.

The results of this comparison illustrate a fundamental point. Although the MLP is highly flexible and achieves excellent accuracy on the training window, its predictions deteriorate rapidly on unseen data, demonstrating the pitfalls of overfitting and the instability of purely data-driven models in autoregressive settings. The linear regression baseline performs moderately but fails to capture the underlying physics, leading to systematic errors. In contrast, the 2R2C model, despite being governed by only a handful of parameters, extrapolates much more consistently. It responds in the correct direction to changes in heating and solar inputs, maintains stable long-term predictions, and provides parameters that map directly to interpretable physical properties such as insulation and thermal mass.  

This example therefore highlights the dual advantages of physics-based modeling: the ability to generalize beyond the training window and the guarantee of action-consistency rooted in thermodynamic reasoning. At the same time, it underscores the limitations of purely data-driven black-box models when asked to predict system behavior under conditions not seen during training.

```{code-cell} ipython3
:tags: [hide-input]
:load: _static/rcnetwork.py
``` -->


