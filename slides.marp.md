---
marp: true
paginate: true
theme: default
class: lead
size: 16:9
headingDivider: 2
math: katex
title: Modeling to Decisions — State-Space, Simulation, and OCP
description: CS/ML grad-friendly deck built around the function+noise viewpoint
---

# Modeling → State-Space → Simulation → OCP

- A 2-hour fast-track, tuned for CS/ML grads
- From modeling discipline to actionable optimization
- Thread: state/control/disturbance/observation + function+noise + kernels

---

## Hook: Why modeling before algorithms?

- Supervised learning is plug-and-play; RL/control need problem definition
- Decisions require: objectives, constraints, time structure, information flow
- We optimize behavior, not just predictions — structure matters

---

## Modeling mindset (from the book intro)

- Model = decision problem specification, not just a predictor
- Define early: objectives, hard constraints, observability, time scale
- Start simple, surface constraints, only add structure when needed

---

## Decision-making models (from Modeling)

- Purpose: roll trajectories under candidate inputs for planning/evaluation
- Inputs: control u; exogenous drivers d; state x; observations y
- Interface: given (x0, {u}, {d}) → trajectory {x, y}

---

## Time representations

- Discrete: t = 0,1,2,... aligns with sensing/actuation and logged data
- Continuous: t ∈ ℝ≥0 matches physics; discretize for computation
- Both views interconvert; pick based on sensing/actuation and clarity

---

## State-space form

- Discrete:  x_{t+1} = f_t(x_t, u_t),   y_t = h_t(x_t, u_t)
- Continuous:  \(\dot{x}(t) = f(x(t), u(t))\),  \(y(t) = h(x(t), u(t))\)
- Linear choice trades fidelity for transparency/speed

\[
\dot{x} = A x + B u, \quad y = C x + D u.
\]

---

## Math seeds: discretization and linearization

- Forward Euler:  \(x_{t+1} \approx x_t + \Delta t\, f(x_t, u_t)\)
- Linearization at (\(\bar{x},\bar{u}\)):  \(f(x,u) \approx f(\bar{x},\bar{u}) + A\,(x-\bar{x}) + B\,(u-\bar{u})\)
- Stochastic: Euler–Maruyama for \(dX_t = f(X_t,U_t)dt + \sigma(X_t,U_t)dW_t\)

\[
X_{t+\Delta t} \approx X_t + f(X_t,U_t)\,\Delta t + \sigma(X_t,U_t)\,\sqrt{\Delta t}\,\varepsilon,\ \varepsilon\sim\mathcal{N}(0,I).
\]

---

## From deterministic to stochastic

- Two equivalent lenses:
  - Function+noise: generative mechanism
  - Transition kernel: distribution view
- Use whichever is most natural/available

---

## Deep dive: Function + Noise

- Additive: \(x_{t+1} = f(x_t,u_t) + w_t\)
- Multiplicative: \(x_{t+1} = f(x_t,u_t) + \Gamma(x_t,u_t)\,w_t\)
- Densities via pushforward/change of variables when \(\Gamma\) invertible

\[
p(x'\mid x,u) = p_w\!\left(\Gamma^{-1}(x,u)[x' - f(x,u)]\right)\,\left|\det\Gamma^{-1}(x,u)\right|.
\]

- Intuition: control moves the mean; \(\Gamma\) shapes dispersion/orientation

---

## Transition kernel view

- Directly specify \(p(x'\mid x,u)\); function+noise is a special case
- Discrete example: \(p(x'\mid x,u) = \sum_i \mathbf{1}\{f(x,u,w_i)=x'\}\,p_i\)
- Useful when we can sample or estimate transitions without mechanism

---

## Partial observability and filtering

- Dynamics: \(x_{t+1}=f(x_t,u_t,w_t)\), Observation: \(y_t=h(x_t,v_t)\)
- Linear-Gaussian special case leads to Kalman filtering
- Kernel form: \(p(y\mid x)\) abstracts sensor mechanism

---

## Deep dive: Kalman predict/update (compact)

- Model: \(x_{t+1}=A x_t + B u_t + w_t\), \(w_t\sim\mathcal{N}(0,Q)\)
- Obs: \(y_t=C x_t + D u_t + v_t\), \(v_t\sim\mathcal{N}(0,R)\)
- Predict: \(\hat{x}_{t|t-1}=A\hat{x}_{t-1}+Bu_{t-1}\), \(P_{t|t-1}=AP_{t-1}A^\top+Q\)
- Gain: \(K_t=P_{t|t-1}C^\top(CP_{t|t-1}C^\top+R)^{-1}\)
- Update: \(\hat{x}_t=\hat{x}_{t|t-1}+K_t(y_t - C\hat{x}_{t|t-1}-Du_t)\)

Intuition: predict then correct; K balances model vs measurement uncertainty.

---

## Example bank (ML-tuned)

- 1D double integrator (robot on a line)
- Inventory control (stochastic demand)
- Queue length (admissions/throttling)

Each: define state, control, disturbance, observation, constraints, and how \(p(x'\mid x,u)\) shifts with u.

---

## Example 1: 1D double integrator

- State: \(x_t = [s_t,\ v_t]^\top\)
- Control: \(u_t\) (acceleration), Disturbance: small process noise
- Discrete dynamics: \(s_{t+1}=s_t+v_t\Delta t\), \(v_{t+1}=v_t+u_t\Delta t + w_t\)
- Constraints: speed/accel bounds; observations could be position only
- Kernel: mean shifts linearly with u; variance from process noise

---

## Example 2: Inventory control (worked kernel)

- State: on-hand inventory \(x_t\); Control: order amount \(u_t\)
- Demand \(d_t\) stochastic; Next state: \(x_{t+1} = x_t + u_t - d_t\)
- If \(d_t\sim \text{Poisson}(\lambda)\):
  - PMF of \(x_{t+1}\) is a shifted Poisson by \(x_t+u_t\)
- Gaussian approximation: \(d_t\sim\mathcal{N}(\mu,\sigma^2)\)
  - \(x_{t+1}\sim\mathcal{N}(x_t+u_t-\mu,\sigma^2)\)
- Takeaway: control shifts the kernel; uncertainty comes from demand

---

## Example 3: Queue length

- State: queue length \(q_t\); Control: admission/throttle \(u_t\in[0,1]\)
- Arrivals \(a_t\), service \(s_t\) stochastic; \(q_{t+1}=\max\{0,\ q_t + a_t(u_t) - s_t\}\)
- Kernel: control modulates arrival rate (e.g., Poisson with rate \(\lambda u_t\))
- Observations: exact count or noisy sensing

---

## Programs as models (Simulation)

- Analytical: expose local dynamics \(f\)
- Simulation: trajectory generator \(\mathcal{S}(x_0,\{u\})\); internal logic hidden
- Useful paradigms: Discrete-Event (DES), Hybrid, Agent-Based (ABM)

---

## DES: event calendar (concept)

- State changes at event times; between events, state is constant
- Components: state set, events, transition function, time-advance

---

## Deep dive: DES queue pseudocode

```python
# Event calendar simulation for a single-server queue
init_state()
now = 0.0
schedule(Event('arrival', t=sample_arrival()))
schedule(Event('departure', t=+inf))

while now < T:
    e = pop_next_event()
    now = e.t
    if e.type == 'arrival':
        q = min(q_max, q + 1)  # admit or throttle based on policy u(now)
        if server_idle:
            server_idle = False
            schedule(Event('departure', t=now + sample_service()))
        schedule(Event('arrival', t=now + sample_arrival()))
    elif e.type == 'departure':
        if q > 0:
            q -= 1
            schedule(Event('departure', t=now + sample_service()))
        else:
            server_idle = True
```

Policy hooks: throttle admissions; prioritize classes; preemptions.

---

## Hybrid systems (guards/resets)

- Modes \(q\in\mathcal{Q}\) with mode-specific \(\dot{x}=f_q(x)\)
- Guards trigger transitions; reset maps update state at switches
- Thermostat example: piecewise dynamics with hysteresis

---

## Agent-Based models (ABM)

- Many agents with local rules; macro behavior emerges (traffic, markets)
- Useful when global transition is hard to write but local rules are clear

---

## Bridge to OCP

- Goal: compute actions over time to optimize a criterion
- Discrete-time Bolza form:
\[
\min_{u_{1:T-1}}\ c_T(x_T) + \sum_{t=1}^{T-1} c_t(x_t,u_t)\quad \text{s.t.}\ x_{t+1}=f_t(x_t,u_t),
\]
- Variants: Lagrange (running only), Mayer (terminal only)
- Equivalence: reduce to Mayer by state augmentation

---

## Deep dive: OCP template (discrete)

- Variables: \(x_{1:T}, u_{1:T-1}\)
- Dynamics/constraints: equality + bounds
- Single vs multiple shooting vs direct collocation — conditioning trade-offs

\[
\begin{aligned}
\min\ & c_T(x_T) + \sum c_t(x_t,u_t)\\
\text{s.t.}\ & x_{t+1}-f_t(x_t,u_t)=0,\\
& u_\ell\le u_t\le u_u,\; g_t(x_t,u_t)\le 0,\; h(x_1,x_T)=0.
\end{aligned}
\]

---

## Deep dive: Drive-to-target with minimum energy (LQR when LQ)

- Double integrator, \(\Delta t=1\): \(s_{t+1}=s_t+v_t\), \(v_{t+1}=v_t+u_t\)
- Cost: \(J=\sum_t \tfrac{\beta}{2}u_t^2 + \tfrac{\gamma}{2}v_t^2\)
- Boundary: \((s_1,v_1)=(0,0)\), \((s_{T+1},v_{T+1})=(S,0)\)
- Linear dynamics + quadratic costs ⇒ LQR structure; else use direct methods

Practical tip: keep states explicit if you have long horizons/constraints.

---

## Wrap

- Modeling upstream of learning makes downstream optimization tractable
- The function+noise and kernel views unify probabilistic modeling
- DES/Hybrid/ABM broaden the modeling toolbox beyond equations
- OCP turns modeled structure into actionable decisions

Thank you!
