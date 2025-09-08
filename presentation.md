---
marp: true
theme: mila
paginate: true
class: lead
fragment: false
math: mathjax

style: |
  .two-columns {
    display: flex;
    gap: 15px;
    margin-top: 2px;
  }
  .column {
    flex: 1;
    border: 1px solid #ccc;
    padding: 15px;
    border-radius: 8px;
    background-color: #f9f9f9;
    font-size: 0.9em;
  }
---

# Practical Reinforcement Learning: From Algorithms to Applications
## Lecture 1: September 3rd, 2025
*Pierre-Luc Bacon*  
Associate Professor – Université de Montréal  

---

## Logistics

- **Time/location**: Mon 12:30 to 14:29, Wed 10:30 to 12:29. 10 mins break after 50 mins
- **Office hours**: Offered by TA. Appointment with me possible via email
- **Website/notes**: pierrelucbacon.com/rlbook
- **Slides**: This deck; updates pushed regularly
- **Discussion**: Moodle/Slack

---

## Prerequisites

- **Math**: Linear algebra, probability, multivariate calculus
- **Optimization**: Basic convex/nonlinear optimization
- **Programming**: Python/NumPy; JAX/PyTorch helpful
- **Control/RL**: Intro-level familiarity is a plus

---

## Math and proofs in this course

- **Yes, there will be math and proofs**, but for insight, not to math-intimidate
- Proofs help us understand how methods relate to one another and probe their limitations
- **You won't be asked to regurgitate proofs** on exams
- Focus: Can you *use* the result? Do you understand the *assumptions*?
- Example: Bellman's principle → why dynamic programming works
- Example: Contraction mapping → when value iteration converges

**Goal**: Mathematical maturity to read papers and debug algorithms

---

## Policy Regarding LLMs

**LLMs are here to stay**: they're powerful tools that will be part of your career for
- Code generation and debugging
- Understanding concepts
- Exploring different approaches

**We won't pretend they don't exist** or use detectors. 


**But this course is your chance to go deep**, to really immerse yourself in the material

**There's no shortcut to genuine understanding**. You need to wrestle with the concepts yourself

**Assessment**: Individual oral interviews where you explain your thought process, walk through solutions, and discuss challenges you encountered

**Why interviews?** Because true learning means you can articulate your understanding, not just produce answers

---

## Grading breakdown

**40% Final exam** - Wednesday, December 10
- No remote exams or alternative dates for travel
- Must be taken in person

**20% Final project** - Monday, December 15 (poster presentation)
- **No remote presentations allowed**
- Reproducible code + case study
- Applied problem: RL and optimal control
- **Not allowed**: MuJoCo, Atari environments, Minigrid. Must be a "real" application
- **No paper to write**: poster presentation only

**20% Midterm** - Monday, October 27 (after fall break)
- In-person exam

**10% Homework** - 2 assignments with oral interviews
- Individual explanations of thought process and solutions
- A1: October 6th to October 20th 
- A2: November 17th to December 1st

---

# Why RL Hasn't Eaten the World (Yet)

**Supervised Learning Success:**
- Standardized tools: scikit-learn, TensorFlow, PyTorch
- Medicine: 100s of FDA-approved ML devices
- Routine integration in production workflows

**Reinforcement Learning Reality:**
- **0** FDA-cleared devices explicitly reference RL (as of 2025)
- Only **2/86** clinical AI trials tested RL-based decisions (*Lancet* 2024)
- Google DeepMind data center cooling (2018): Limited public confirmation since

$$\text{The Gap: } \underbrace{\text{Impressive Demos}}_{\text{AlphaGo, Atari}} \neq \underbrace{\text{Deployed Systems}}_{\text{Where is RL?}}$$

---

# The Real Bottleneck

> *"The biggest constraint on progress is not limited computer power, but instead the difficulty of learning the underlying structure of the decision problem."*
> — John Rust, 2008

**It's not about algorithms.** We have:
- Temporal difference methods
- Policy gradients  
- Model-based planning
- Hierarchical RL

**It's about problem formulation:**
- What are we optimizing?
- What constraints are non-negotiable?
- What information is available, and when?
- How do tradeoffs work in practice?

---

# The Modeling Gap

**RL's Comfortable Abstraction:**
```python
env = gym.make("CartPole-v1")
state = env.reset()
action = policy(state)
next_state, reward, done = env.step(action)
```

**Reality Check:**
- Sensors produce noisy, partial data
- Constraints: safety limits, budgets, regulations
- Objectives conflict and shift
- Time structure isn't given

**The Bridge:** From messy reality → structured decision problems

This is modeling. And it's what this course is about.

---

# Models for Decision-Making

**Not just prediction machines, but trajectory generators:**

$$(\mathbf{x}_0, \{\mathbf{u}_t\}_{t=0}^{T-1}, \{\mathbf{d}_t\}_{t=0}^{T-1}) \longmapsto \{\mathbf{x}_t, \mathbf{y}_t\}_{t=0:T}$$

- $\mathbf{u}_t$: **controls** we choose (heating power, irrigation, routing)
- $\mathbf{d}_t$: **exogenous drivers** we don't control (weather, demand)
- $\mathbf{x}_t$: **system state** (temperatures, water levels, queue lengths)
- $\mathbf{y}_t$: **observations** (sensor readings)

**Two key requirements:**
1. **Responsiveness to inputs** - expose the levers that matter
2. **Compact memory** - state $\mathbf{x}_t$ summarizes "what matters so far"

---

## World models

- Today, we often call these dynamics models "world models"
- Not limited to vision or VLMs; modality-agnostic (text, sensors, audio, video)
- What makes them unique here: they answer counterfactuals under actions
  - If we apply input sequence $\{\mathbf{u}_t\}$, what plausible trajectory follows?
- Goal: generate trajectories that are not just likely, but physically/plausibly consistent

Examples:
- HVAC: How does temperature evolve if we change heating policy?
- Hydro: How do levels respond under different turbine schedules and inflow regimes?
- Robotics/VLMs: Plan sequences conditioned on visual state but still require dynamics that respect constraints

---

# State-Space Models

The most common way we'll model dynamics is in state-space (what EE folks call it); this is natural for CS, though other representations (e.g., frequency-domain) also exist and are useful.

**Discrete Time:**
$$\mathbf{x}_{t+1} = f_t(\mathbf{x}_t, \mathbf{u}_t), \quad \mathbf{y}_t = h_t(\mathbf{x}_t, \mathbf{u}_t)$$

**Continuous Time:**
$$\dot{\mathbf{x}}(t) = f(\mathbf{x}(t), \mathbf{u}(t)), \quad \mathbf{y}(t) = h(\mathbf{x}(t), \mathbf{u}(t))$$

**The parallel to RNNs:**
- State = hidden vector
- Control = input sequence  
- Output = read-out layer
- Dynamics = recurrence relation

Digital controllers sample in discrete steps; physics evolves continuously.
Both views are needed, connected via DACs and ADCs.

---

## Continuous-time dot notation

- The dot over a variable denotes a time derivative: $\dot{\mathbf{x}}(t) = d\mathbf{x}/dt$
- It encodes the instantaneous rate of change as time advances in infinitesimal steps $dt$
- Ordinary differential equations (ODEs) specify dynamics: $\dot{\mathbf{x}}(t) = f(\mathbf{x}(t), \mathbf{u}(t))$
- Discrete-time models arise by integrating over a small step $\Delta t$ (e.g., Euler):
  $$\mathbf{x}_{t+1} \approx \mathbf{x}_t + \Delta t\, f(\mathbf{x}_t, \mathbf{u}_t)$$
- We'll mostly use ODE-based state-space models in this course

Note: Other formalisms exist (PDE control for spatial fields, SDEs for stochastic dynamics), but we will not cover them here.

---

# Linear Systems

$$\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u} + E\mathbf{d}, \quad \mathbf{y} = C\mathbf{x} + D\mathbf{u}$$

**This is NOT a claim that the world is linear!**

It's a modeling choice:
- **Pro:** Fast simulation, closed-form analysis, guaranteed stability
- **Pro:** Interpretable parameters (thermal resistance, capacitance)
- **Pro:** Works with limited data
- **Con:** Limited expressiveness
- **Con:** May need piecewise models

**Deep learning note:** Recent "state space" sequence models (e.g., S4/DSS) revive this view.
- They start from linear state-update cores but interleave nonlinearities between layers/blocks
- The linear core ≠ a linear network; expressivity comes from stacking and nonlinear readouts

*"Linearity is not a belief about the world, it is a modeling choice that trades fidelity for transparency and speed."*

---

# Case Study: Montréal in February (-20°C)

**Single thermal mass model:**
$$\dot{T}(t) = -\frac{1}{RC}T(t) + \frac{1}{RC}T_{\text{out}}(t) + \frac{1}{C}u(t)$$

**Physical interpretation:**
- $T$: indoor temperature (state)
- $u$: heating power in watts (control)
- $T_{\text{out}}$: outdoor temperature (disturbance)
- $R$: thermal resistance (insulation quality)
- $C$: thermal capacitance (energy to heat the air)

**Standard form:** $\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u} + E\mathbf{d}$

$$A = -\frac{1}{RC}, \quad B = \frac{1}{C}, \quad E = \frac{1}{RC}$$

---

# Walls Have Memory: Two-State Model

**Energy balance equations:**
$$C_{\text{air}}\frac{dT_{\text{in}}}{dt} = \frac{T_{\text{wall}} - T_{\text{in}}}{R_{ia}} + u(t)$$
$$C_{\text{wall}}\frac{dT_{\text{wall}}}{dt} = \frac{T_{\text{out}} - T_{\text{wall}}}{R_{wo}} - \frac{T_{\text{wall}} - T_{\text{in}}}{R_{ia}}$$

**Matrix form with $\mathbf{x} = [T_{\text{in}}, T_{\text{wall}}]^T$:**

$$A = \begin{bmatrix}
-\frac{1}{R_{ia}C_{air}} & \frac{1}{R_{ia}C_{air}} \\
\frac{1}{R_{ia}C_{wall}} & -(\frac{1}{R_{ia}}+\frac{1}{R_{wo}})\frac{1}{C_{wall}}
\end{bmatrix}$$

Each element has meaning: $A_{12}$ = heat from wall to air, $A_{21}$ = heat from air to wall

---

# Why Not Just Neural ODEs?

**Neural ODE approach:**
$$\dot{\mathbf{z}}(t) = f_{\boldsymbol{\theta}}(\mathbf{z}(t), \mathbf{u}(t), \mathbf{d}(t))$$

**RC Models Win When:**
- Limited data (days not months)
- Parameters map to physics (can validate against building specs)
- Need to modify (what if we insulate this wall?)
- Fast simulation required (linear is cheap)
- Interpretability matters (which zone causes slow heating?)

**Neural ODEs Win When:**
- Abundant data available
- Complex nonlinearities dominate
- No domain knowledge
- Pure prediction accuracy matters

**Best Practice:** RC backbone + learned residual corrections

---

# From Deterministic to Stochastic

**Two equivalent views:**

**1. Function + Noise (Constructive)**
$$\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t) + \mathbf{w}_t, \quad \mathbf{w}_t \sim p_{\mathbf{w}}$$
- Explicit noise source
- Can track variability along trajectories
- Enables reparameterization tricks

**2. Transition Kernel (Abstract)**
$$p(\mathbf{x}_{t+1}|\mathbf{x}_t, \mathbf{u}_t)$$
- More general (includes non-additive noise)
- Natural for RL formulations
- Don't need to specify noise structure

**Continuous time:** SDEs with drift and diffusion
$$d\mathbf{X}_t = f(\mathbf{X}_t, \mathbf{U}_t)dt + \sigma(\mathbf{X}_t, \mathbf{U}_t)d\mathbf{W}_t$$

---

## From function+noise to kernel (derivation)

Start with generative dynamics:
$$\mathbf{x}_{t+1} = f(\mathbf{x}_t,\mathbf{u}_t,\mathbf{w}_t),\quad \mathbf{w}_t\sim p_{\mathbf{w}}(\cdot)$$

The induced transition kernel is the pushforward of $p_{\mathbf{w}}$:
$$p(\mathbf{x}_{t+1}\mid \mathbf{x}_t,\mathbf{u}_t) = \int \delta\!\big(\mathbf{x}_{t+1} - f(\mathbf{x}_t,\mathbf{u}_t,\mathbf{w})\big)\, p_{\mathbf{w}}(\mathbf{w})\, d\mathbf{w}$$

Special cases:
- Additive noise: $\;\mathbf{x}_{t+1} = f(\mathbf{x}_t,\mathbf{u}_t)+\mathbf{w}_t$
  $$p(\mathbf{x}_{t+1}\mid \mathbf{x}_t,\mathbf{u}_t) = p_{\mathbf{w}}\!\big(\mathbf{x}_{t+1} - f(\mathbf{x}_t,\mathbf{u}_t)\big)$$
- Affine noise: $\;\mathbf{x}_{t+1} = f(\mathbf{x}_t,\mathbf{u}_t)+ \Gamma(\mathbf{x}_t,\mathbf{u}_t)\,\mathbf{w}_t$ with invertible $\Gamma$
  $$p(\mathbf{x}_{t+1}\mid \mathbf{x}_t,\mathbf{u}_t) = p_{\mathbf{w}}\!\big(\Gamma^{-1}(\mathbf{x}_t,\mathbf{u}_t)[\mathbf{x}_{t+1}-f]\big)\, \big|\det\Gamma^{-1}\big|$$
- Discrete noise: $\;\mathbf{w}_t\in\{w_i\}$ with probs $p_i$
  $$p(\mathbf{x}_{t+1}\mid \mathbf{x}_t,\mathbf{u}_t) = \sum_i \mathbb{1}\{f(\mathbf{x}_t,\mathbf{u}_t,w_i)=\mathbf{x}_{t+1}\}\, p_i$$

Interpretation: the kernel marginalizes the latent noise; the function+noise view is constructive, the kernel view is abstract but more general.

---

# Real Example: Québec's Giant Battery

**Robert-Bourassa Reservoir:**
- 62 km³ of water (> Lake Ontario's active volume)
- 5.6 GW capacity (1/5 of Hydro-Québec's total)
- Powers aluminum smelters (voltage dips = millions lost)

**Mass balance dynamics:**
$$\mathbf{x}_{t+1} = \mathbf{x}_t + \mathbf{r}_t - \mathbf{u}_t$$

**Stochastic inflow model:**
$$\mathbf{w}_t \sim \begin{cases}
0 & \text{prob } p_0 \text{ (no rain)} \\
\text{LogNormal}(\mu, \sigma^2) & \text{prob } 1-p_0 \text{ (storm/melt)}
\end{cases}$$

Spring: snowmelt → log-normal tails
Summer: convective storms → point mass at 0 + heavy tail

---

# Partial Observability: Hidden States

**Full system:**
$$\begin{aligned}
\mathbf{x}_{t+1} &= f_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t), \quad \mathbf{w}_t \sim p_{\mathbf{w}} \\
\mathbf{y}_t &= h_t(\mathbf{x}_t, \mathbf{v}_t), \quad \mathbf{v}_t \sim p_{\mathbf{v}}
\end{aligned}$$

**You never see the true state!**
- State evolves with process noise $\mathbf{w}_t$
- Observations corrupted by measurement noise $\mathbf{v}_t$

**Linear-Gaussian case (Kalman filter territory):**
$$\begin{aligned}
\mathbf{x}_{t+1} &= A\mathbf{x}_t + B\mathbf{u}_t + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(0, Q) \\
\mathbf{y}_t &= C\mathbf{x}_t + D\mathbf{u}_t + \mathbf{v}_t, \quad \mathbf{v}_t \sim \mathcal{N}(0, R)
\end{aligned}$$

---

# Example: Adaptive Optics for Telescopes

**Problem:** Atmosphere distorts starlight in milliseconds

**State:** Wavefront distortion coefficients $\mathbf{x}_t \in \mathbb{R}^n$

**Dynamics:** Wind-blown turbulence
$$\mathbf{x}_{t+1} = \mathbf{A}\mathbf{x}_t + \mathbf{w}_t$$
- $\mathbf{A}$ shifts pattern spatially
- $\mathbf{w}_t$ follows Kolmogorov spectrum (low frequencies dominate)

**Observations:** Can't see wavefront, only slopes!
$$\mathbf{y}_t = \mathbf{C}\mathbf{x}_t + \boldsymbol{\varepsilon}_t$$
- $\mathbf{C}$ maps distortion → measurable gradients
- $\boldsymbol{\varepsilon}_t$ is photon noise

**Control:** Deformable mirror correction
$$\mathbf{x}_t^{\text{residual}} = \mathbf{x}_t - \mathbf{B}\mathbf{u}_t$$

Must decide $\mathbf{u}_t$ every millisecond using noisy gradient measurements!

---

# Programs as Models: Beyond Equations

**Analytical Models:** Give us $f$ directly
$$\mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k)$$

**Simulation-Based Models:** Black-box trajectory generator
$$\{\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_T\} = \mathcal{S}(\mathbf{x}_0, \{\mathbf{u}_t\}_{t=0}^{T-1})$$

**Examples:**
- **MuJoCo:** Solves $M\ddot{q} + C = \tau + J^T\lambda$ with contact constraints
- **EnergyPlus:** 100k+ lines mixing heat equations + schedules + state machines
- **SUMO:** Each vehicle is an agent following car-following rules

The simulator encapsulates details we can't easily write as $f$.

---

## Simulation-based modeling vs closed-form densities

- Sometimes it's easier to write a program that samples next states than to write a closed-form $p(\mathbf{x}_{t+1}\mid \mathbf{x}_t,\mathbf{u}_t)$
- The simulator is a black box: we provide $(\mathbf{x}_t,\mathbf{u}_t)$, it returns a sample $\mathbf{x}_{t+1}$ (and an observation)
- The transition "model" is implicitly defined by the code, not an equation

Implications:
- We can generate counterfactual trajectories under candidate inputs
- We learn/validate using samples, without needing analytic densities
- Often only outputs are observable; inner workings are hidden by design

---

# Discrete Events & Hybrid Systems

**Discrete-Event Systems:** State changes only at events
- States: $\mathcal{X}$ (network routing tables)
- Events: $\mathcal{E}$ (packet arrival, link failure)
- Transition: $f: \mathcal{X} \times \mathcal{E} \rightarrow \mathcal{X}$

**Hybrid Systems:** Continuous dynamics + discrete modes
```python
if temp < setpoint - delta:
    mode = "heating"
    dx/dt = f_heat(x)
elif temp > setpoint + delta:  
    mode = "cooling"
    dx/dt = f_cool(x)
else:
    mode = "off"
    dx/dt = f_off(x)
```

Guards trigger transitions; reset maps update state.

---

# Agent-Based Models: Emergence from Individuals

**Neighborhood Energy under Dynamic Pricing:**
```python
for household in neighborhood:
    price = utility.get_current_price()
    optimal_setpoint = household.mpc_controller.optimize(
        current_temp=household.temperature,
        price_forecast=utility.price_forecast,
        comfort_weight=household.preference
    )
    household.set_hvac_setpoint(optimal_setpoint)
    neighborhood.total_demand += household.power_consumption
```

**No global equation!** Macro patterns (peak shifting, rebounds) emerge from individual optimization.

**SUMO Traffic:** Each vehicle follows Krauss rule
$$v_{\text{safe}} = v_{\text{leader}} + \frac{\text{gap} - \text{mingap}}{\tau}$$

Thousands of local decisions → traffic waves, bottlenecks
---

# Discrete-Time Trajectory Optimization (Open-Loop)

**Trajectory = states + controls over time**

$$
\text{trajectory } = \big(\{\mathbf{x}_t\}_{t=0}^{T},\;\{\mathbf{u}_t\}_{t=0}^{T-1}\big)
$$

- Plan the entire control sequence in advance (open loop) and apply it as-is.
- Pro: clean, finite-dimensional program in discrete time; exposes structure.
- Limitation: no feedback → disturbances/model error can compound.
- Continuous time is infinite-dimensional; direct methods discretize/parameterize.

---

# Examples of DOCPs (native discrete-time)

- Inventory control (periodic orders):
  $$x_{k+1}=x_k+u_k-d_k,\quad c_k=h[x_k]_+ + p[-x_k]_+ + c\,u_k$$

- End-of-day portfolio rebalancing:
  $$h_{k+1}=(h_k+u_k)\odot(\mathbf{1}+\mu_k)$$

- Daily ad-budget with carryover:
  $$s_{k+1}=\alpha s_k+\beta u_k,\quad \max\sum_k g(s_k,u_k)-c\,u_k$$

---

# From ODEs to DOCPs via a one-step map

Continuous-time dynamics:

$$
\dot{\mathbf{x}}(t)=\mathbf{f}(t,\mathbf{x}(t),\mathbf{u}(t)),\qquad \mathbf{x}(0)=\mathbf{x}_0.
$$

With step $\Delta$ and grid $t_k=k\Delta$:

$$
\mathbf{x}_{k+1}=\mathbf{F}_\Delta(\mathbf{x}_k,\mathbf{u}_k,t_k)
$$

Bolza-form DOCP:

$$
\begin{aligned}
\min_{\{\mathbf{x}_k,\mathbf{u}_k\}}\; & c_T(\mathbf{x}_T)+\sum_{k=0}^{T-1} c_k(\mathbf{x}_k,\mathbf{u}_k)\\
\text{s.t.}\; & \mathbf{x}_{k+1}-\mathbf{F}_\Delta(\mathbf{x}_k,\mathbf{u}_k,t_k)=0,\quad \mathbf{x}_0=\mathbf{x}_\mathrm{init}.
\end{aligned}
$$

---

# Programs as DOCPs

Program as a dynamical system with state and controls:

$$
\mathbf{x}_{k+1}=\Phi_k(\mathbf{x}_k,\mathbf{u}_k),\qquad
\min_{\{\mathbf{u}_k\}}\; c_T(\mathbf{x}_T)+\sum_{k=0}^{T-1} c_k(\mathbf{x}_k,\mathbf{u}_k).
$$

- Differentiable programming (JAX/PyTorch): reverse-mode AD for gradients.
- Non-differentiable pieces: derivative-free/weak-gradient (FD, SPSA, CMA-ES), with smoothing if needed.

---

## Example: HTTP retrier with backoff

- State: $(t, k, \text{done}, \text{code}, \text{jitter})$; Control: wait $u_k$.
- Transition: wait + probabilistic request outcome.
- Objective: penalize latency and failure; optimize backoff schedule.

---

## Example: Gradient descent with momentum

Loss:

$$
\ell(\boldsymbol{\theta})=\tfrac{1}{2}\,\lVert \mathbf{A}\boldsymbol{\theta}-\mathbf{b}\rVert_2^2,\quad
\mathbf{x}_k=\begin{bmatrix}\boldsymbol{\theta}_k\\ \mathbf{m}_k\end{bmatrix},\;\mathbf{u}_k=\begin{bmatrix}\alpha_k\\ \beta_k\end{bmatrix}.
$$

Step:

$$
\Phi_k(\mathbf{x}_k,\mathbf{u}_k)=\begin{bmatrix}
\boldsymbol{\theta}_k-\alpha_k(\beta_k\mathbf{m}_k+\mathbf{A}^\top(\mathbf{A}\boldsymbol{\theta}_k-\mathbf{b}))\\[2mm]
\beta_k\mathbf{m}_k+\mathbf{A}^\top(\mathbf{A}\boldsymbol{\theta}_k-\mathbf{b})
\end{bmatrix}.
$$

Hyperparameter optimization as DOCP:

$$
\min_{\{\alpha_k,\beta_k\}}\; \ell(\boldsymbol{\theta}_T)+\sum_k\big(\rho_\alpha\alpha_k^2+\rho_\beta(\beta_k-\bar\beta)^2\big).
$$

---

## Backprop = reverse-time costate

$$
\boldsymbol{\lambda}_T=\nabla_{\mathbf{x}_T} c_T,\qquad
\boldsymbol{\lambda}_k=\nabla_{\mathbf{x}_k} c_k+\big(\nabla_{\mathbf{x}_k}\Phi_k\big)^\top\boldsymbol{\lambda}_{k+1},\quad
\nabla_{\mathbf{u}_k}\mathcal{J}=\nabla_{\mathbf{u}_k} c_k+\big(\nabla_{\mathbf{u}_k}\Phi_k\big)^\top\boldsymbol{\lambda}_{k+1}.
$$

Exactly what reverse-mode autodiff computes.

---

# The Discrete-Time Optimal Control Problem

**Canonical Bolza Form:**
$$\begin{aligned}
\min_{\mathbf{u}_{1:T-1}} \quad & c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t, \mathbf{u}_t) \\
\text{s.t.} \quad & \mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t), \quad t = 1,\dots,T-1 \\
& \mathbf{x}_1 \text{ given}
\end{aligned}$$

**Special Cases:**
- **Lagrange:** Only running costs $\sum_t c_t(\mathbf{x}_t, \mathbf{u}_t)$
- **Mayer:** Only terminal cost $c_T(\mathbf{x}_T)$

**Key Challenge:** Dimensionality grows with horizon
- $(T-1)(n+m)$ variables
- Temporal coupling through dynamics

---

# State Augmentation: Everything is Mayer

**Transform Bolza → Mayer by tracking cumulative cost:**

Define augmented state:
$$\tilde{\mathbf{x}}_t = \begin{bmatrix} \mathbf{x}_t \\ y_t \end{bmatrix}$$

Augmented dynamics:
$$\tilde{\mathbf{x}}_{t+1} = \begin{pmatrix}
\mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t) \\
y_t + c_t(\mathbf{x}_t, \mathbf{u}_t)
\end{pmatrix}$$

New terminal cost:
$$\tilde{c}_T(\tilde{\mathbf{x}}_T) = c_T(\mathbf{x}_T) + y_T$$

**Why?** Simplifies derivations, unifies implementation.

---

# Eco-Cruise: Complete Example

**Problem:** Drive 1 km in 60 seconds, minimize energy

**State & Dynamics:**
$$s_{t+1} = s_t + v_t, \quad v_{t+1} = v_t + u_t$$
- $s_t$: position, $v_t$: velocity, $u_t$: acceleration

**Cost:** Balance acceleration and speed penalties
$$c_t(v_t, u_t) = \frac{\beta}{2}u_t^2 + \frac{\gamma}{2}v_t^2$$
- $\beta = 1.0$ (acceleration penalty)
- $\gamma = 0.1$ (speed penalty)

**Constraints:**
- $0 \leq v_t \leq 20$ m/s (speed limits)
- $|u_t| \leq 3$ m/s² (comfort)
- $s_1 = 0, v_1 = 0, s_{61} = 1000, v_{61} = 0$

---

# Solution Methods: Three Approaches

**1. Direct (Simultaneous):** All states and controls as variables
```python
z = [s₁, v₁, u₁, s₂, v₂, u₂, ..., s_T, v_T, u_{T-1}]
minimize f(z) subject to dynamics_constraints(z)
```

**2. Single Shooting:** Eliminate states via simulation
```python
def loss(u):
    x = x₀
    for t in range(T):
        cost += c(x, u[t])
        x = f(x, u[t])
    return cost
```

**3. Multiple Shooting:** Hybrid for stability
- Break into K segments
- State variables at segment boundaries
- Simulate within segments

Trade-off: Problem size vs. conditioning

---

# Single Shooting = Deep Learning

**The parallel is exact:**

```python
# Single Shooting OCP          # Neural Network Training
controls = initialize()         params = initialize()
                               
def compute_cost(controls):    def compute_loss(params):
    state = initial_state          hidden = initial_hidden
    cost = 0                       loss = 0
    for t in range(T):            for t in range(T):
        cost += c(state, u[t])        loss += L(hidden, x[t])
        state = f(state, u[t])         hidden = f(hidden, x[t], params)
    return cost                    return loss
    
gradient = autograd(cost)      gradient = autograd(loss)
controls = optimizer.step()    params = optimizer.step()
```

Same tools: Adam, L-BFGS, JAX, PyTorch. "Backprop through the world model."

---

# The Path Forward

**What we're building toward:**

1. **Model Predictive Control (MPC):** Replan at each timestep
2. **Dynamic Programming:** Value functions and Bellman equations  
3. **Learning from Data:** When models aren't given
4. **Learning from Humans:** RLHF meets control

**The mindset shift:**
- Stop treating problems as given (gym.make won't save you)
- Modeling IS the hard part
- Structure enables learning from limited data
- Real impact requires real constraints

**Next:** From trajectories to policies - how do we go from open-loop plans to closed-loop control?

---

<!-- _class: lead -->

# Questions?

**Remember:** 
*"The difficulty is not in solving the problem,*
*but in learning the underlying structure of the decision problem."*

