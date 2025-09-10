---
marp: true
theme: mila
paginate: true
class: lead
fragment: false

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

# RLC 2025 Talk  
### Beyond the Gym Interface
*Pierre-Luc Bacon*  
Associate Professor – Université de Montréal  
*29-05-2025*

---

# Introduction

<div class="two-columns">
  <div class="column">
    <img src="2024milastreet1000x1000.jpg.webp" alt="Mila" style="width: 100%; height: 120px; object-fit: cover; object-position: center 30%; margin-bottom: 15px; display: block;">
    <strong>Mila</strong><br>
    Research institute founded by Université de Montréal and McGill University to advance AI research and foster academic–industry collaboration.
  </div>
  <div class="column">
    <img src="ivadolabs-ezgif.com-jpg-to-png-converter.png" alt="IVADO Labs" style="width: 100%; height: 120px; object-fit: cover; margin-bottom: 15px; display: block;">
    <strong>IVADO Labs</strong><br>
    Nonprofit organization applying AI research to industry problems, focusing on supply chain and operations for Canadian industry.
  </div>
</div>

---

# Tooling Isn't the Bottleneck Anymore

Today's RL libraries are as polished as `sklearn` was for SL.  
So why isn't adoption mainstream?

- The hard part is **upstream of the algorithm**: defining the decision contract.
- Objectives ≠ rewards: true KPIs are negotiated; constraints are real.
- Evidence ≠ a curve: OPE is a smoke test; real impact needs A/B, DiD, synthetic control.
- Environments ≠ Atari: new topologies, new buildings, governance matters.

> *John Rust (2019):  
> "The biggest constraint on progress is not limited computer power,  
> but the difficulty of learning the underlying structure of the decision problem."*

---

# But the RL problem is defined, right?

**Answer:** It depends who you ask.

- If you mean the Suttonian long-term program: yes, the problem is defined. It's an aspiration.
- But in practice, the **world is not given**; there's no firehose of safe experience.

We must **engineer the interface**:

- Define goals, constraints, observability, cadences.
- Accept rewards as surrogates; audit outcomes separately.
- Keep humans in the loop; staged exploration with safety guardrails.
- Prove impact via experiments, not just learning curves.


If the interface is given, **algorithms are the** frontier.
If the interface is not given, **modeling is the frontier**.

---

# The Decision Contract  
### A Model Card for Decisions

**What you must specify before learning anything:**

- **W** — Windows of Control: sensing, decision, actuation cadences.
- **R** — Reward & Risk: objective decomposition, constraints, audit plan.
- **A** — Actions & Actuators: knobs, safety bounds, human-in-loop roles.
- **P** — Platform & Pipeline: data freshness, logging, digital twin.
- **P** — Policies & Priors: baselines (MPC, heuristics), role of learning.
- **E** — Evaluation: experimental design, OPE as smoke test.
- **R** — Real-World Governance: UX, explainability, rollback plan.

---

# Managing Hydroelectric Dams  

Real-world deployment at IVADO Labs:

- Manage private hydroelectric dams for a large metal and mining company.
- Provide **daily decisions** on outflow rates for each dam.
- Objective: minimize power production costs and ensure safety.
- Scale: 1000s of hm³ storage, 100s of m³/s flows.

---

# W — Windows of Control

Our solution is online and realtime, but not in the typical sense.

<img src="window_control.png" alt="Window Control" style="width: 80%; max-width: 600px; height: 300px; display: block; margin: 0 auto 5px auto; object-fit: cover; object-position: center;">

- **Humans must be in the loop**: Commands may require phone calls, manual intervention, and physical dispatch
- **Water movement takes 1-3 days** → decisions must anticipate delays


---

# Rules of The Game

Safety is **non-negotiable**: violating constraints → floods, damage, shortages.

Operational rules:

- Reservoir volumes
- Turbine capacity
- Upstream–downstream delays
- Power balance (including auxiliary power)

---

# Typical Problem Structure

- **Inflow = outflow**: simple linear dynamics, but uncertainty exists (precipitation).
- SDP and multi-scenario MPC work but don't scale; LP solvers need convexification of the costs/constraints.

$$\begin{align}\min_{u_{1:T}} \quad & \sum_{t=1}^T \left[ C_{power}(t) + C_{penalty}(t) \right] \\\text{s.t.} \quad & V_{t+1} = V_t + \Delta t \cdot (I_t - A \cdot u_t), \quad \forall t \in \{1, \ldots, T-1\} \\& V_{min} \leq V_t \leq V_{max}, \quad \forall t \in \{1, \ldots, T\} \\& 0 \leq u_t \leq u_{max}(h_t, MHT_t), \quad \forall t \in \{1, \ldots, T\} \\& u_{min}(t) \leq u_t \leq u_{max}(t), \quad \forall t \in \{1, \ldots, T\} \\& P_{hydro,t} + P_{HQ,t} + P_{gas,t} + P_{deficiency,t} = Load_t, \quad \forall t \in \{1, \ldots, T\} \\& 0 \leq P_{HQ,t} \leq P_{HQ,max}(t), \quad \forall t \in \{1, \ldots, T\} \\& 0 \leq P_{gas,t} \leq P_{gas,max}, \quad \forall t \in \{1, \ldots, T\} \\& P_{deficiency,t} \geq 0, \quad \forall t \in \{1, \ldots, T\} \\& V_1 = V_{initial} \\& u_0 = u_{initial}\end{align}$$


---

#  Nested Cost Structure

The cost function is itself a DP. The "outer agent" tefore chooses the plan flows, and the "inner" one allocates turbines. We can think of HRL. 

$$\begin{align}
\max_{\{q_j\}} \quad & \sum_j P_j(h_j, q_j, \text{turbine\_type}_j) \\
\text{s.t.} \quad & \sum_j q_j = Q_{\text{total}} \\
& q_{j,min}(h_j, \text{turbine\_type}_j) \leq q_j \leq q_{j,max}(h_j, \text{turbine\_type}_j) \\
& P_{j,min}(h_j, \text{turbine\_type}_j) \leq P_j(h_j, q_j, \text{turbine\_type}_j) \leq P_{j,max}(h_j, \text{turbine\_type}_j) \\
& h_j = h_{gross} - h_{losses}(Q_{total}) - h_{aval}(\text{plant\_specific}) \\
& q_{j,max}(h_j) = \min\left(q_{turbine,max}(h_j), q_{cavitation}(h_j), q_{safety}(h_j)\right) \\
& \text{plant\_specific\_constraints}(h_j, q_j, \text{downstream\_levels})
\end{align}$$

This is a sequential allocation problem: how much power to allocate to each turbine. Takes into account turbine capacity curves from the manufacturer, vibration limits, etc. 

---

# Taming Reward Complexity with a Surrogate

Solving a nested DP, too expensive if we want to take lots of rollout. Therefore, we prefer instead learning a surrogate model of the form:

$$\hat{P}_{\text{total}} = f_\theta(V, Q, \text{MHT}) \quad \text{where } f_\theta \approx \text{DP Output}$$

**What this allows us to do:**
- **Massive speedup:** from 10³ to 10⁶+ simulations.
- **Differentiable end-to-end:** policy gradients without implicit differentiation.
- **Keeps physics fidelity:** surrogate trained on DP-generated data.

This is similar to the idea of "best response functions" methods in bilevel optimization. 

(Btw, its also fully jitted thanks to JAX)

---

# Learning Fast Policies with Safety Fallbacks

**Previous setup:**

- Multi-scenario MPC solves a sequence of very large LPs every day.
- We evaluate the scenarios over 70 years of inflow
- It has to be ready by the morning!

**New approach:**

- We "amortize" this expensive computation in a parametrized policy: 

  $$u_t = \pi_\theta(s_t)$$

  instead of re-solving LPs each time.
- This then runs in **milliseconds** at deployment.

But how do we enforce the operational constraints? 

---

# But the policy must stay in the feasible region

**Operational constraints we must respect:**

$$\begin{align}
&u_{\min} \le u_t \le u_{\max}(h_t), \\
&u_t \le u_{\text{cap}}(h_t), \; u_t \le u_{\text{cav}}(h_t), \\
&s_{\min} \le s_t \le s_{\max}, \\
&P_{\text{hydro}}(u_t,h_t)+P_{\text{aux}}\ge P_{\text{demand}}.
\end{align}$$

**We enforce them with:**
- **Action parameterization**: $u^{\text{raw}}_t \xrightarrow{\sigma(\cdot)} u_t' \in [u_{\min}, u_{\max}(h_t)]$
- **Safety-layer projection**: $\tilde u_t = \Pi_{\mathcal K(h_t)}(u_t'),\;\; \mathcal K(h_t)=\{\text{capacity \& cavitation limits}\}$
- **Fast feasibility check** before sending commands:
  - If passes → apply $\tilde u_t$.
  - Else → fallback to MPC action.

We still keep the MPC alive to mitigate deployment risk, and fallback when the fast RL policy is infeasible. Very important for change management too! 

---

# Bootstrap Learning via Behavioral Cloning. 

We **pre-train** our policy on historical data to bootstrap learning in the right region: **Behaviorcal Cloning**
- This also gives us a baseline which is useful for our KPIs

- While our BC policy performs remarkably in terms of RMSE, we can't trust SL alone to generalize safely everywhere, hence the importance of RL + MPC fallback. 
  - We can train on human operator data
  - On the output of the MPC controller
  - Balance deviation using a KL term. 
- Another advantage of starting with BC is that it informs us of design choices down the line when choosing the policy parametrization

---

# Policy Parameterization and Feature Engineering

Representation learning alone isn’t enough here. Unlike image-based RL, there is no raw “pixel input.”
The state must capture physical realities and operational rules that can’t emerge from scratch via deep features.

**Multi-Output MLP**: Residual connections with Swish activation and LayerNorm

**Feature Engineering (249 total):**
- **Temporal**: Cyclical month/day encoding, 1-10 day lags, rolling means
- **Physical Constraints (57)**: Capacity utilization, overflow/shortage risk
- **Water Levels (16)**: Converted from volumes, daily changes, rolling averages  
- **Flow Constraints (21)**: Max capacity, utilization, headroom, turbine groups
- **Predictions**: Daily/weekly/monthly inflow forecasts
- **Coordination**: Upstream lagged outflows (1-3 day delays)

---

# Takeaways

 Real-world impact comes from thoughtful integration, not from pushing new methods everywhere.

 - Decision tools already work: Industry partners have robust processes and expertise.
 - Collaboration matters: Success came from building on what exists, not replacing it.
 - RL is a complement, not a replacement: Adds value where amortization and speed matter.
 - Start from reality: Understand constraints, operator workflow, and risk tolerance first.

AI adoption is about fit: The best projects begin where there's already domain maturity.


---

# RL in HVAC 

HVAC control is everywhere: small improvements scale to massive energy and cost savings.

- HVAC accounts for **40–50% of building energy use**, making it one of the largest energy consumers in the built environment.
- Current systems already use control, but mostly **rule-based or PID** with some MPC in advanced setups.

These methods work, yet they struggle with:

- High variability: Weather, occupancy, internal gains.
- Multiple timescales: Seconds for fans, hours for temperature, days for envelope.
- Complex dynamics: Nonlinear, coupled, with hard constraints.

**Why RL?**
 Potential to adapt online, learn from data, and optimize across conflicting objectives: comfort, energy, cost, emissions.  

--- 

# What I Call RL Here

- This is **RL in spirit**, but not what you see in benchmarks: No PPO, no DQN, no temporal-difference targets.
- The core idea: **Learn from experience**, **plan using that knowledge**, **improve decision-making over time**
- Why this matters: Real systems like HVAC have **hard constraints**, physics, and safety rules. These shape the "game" we play.
- So before diving in, remember: **RL is a paradigm, not a recipe.**

---

# Why Naive RL Fails in HVAC

- **Simulator‑first fallacy**: Building accurate digital twins takes months — material types, occupancy behavior, thermal physics. Many retrofit projects lack historical logs.
- **Black‑box learning is costly**: Requires vast quantities of data and has poor sample efficiency.
- **Unsafe exploration**: Comfort breaches and energy wastage are unacceptable. Live buildings cannot be used for random policy trials.
- **No structure → slow learning**: Black‑box RL relearns basic thermodynamics from scratch. Comfort is fuzzy; reward design is challenging.

**Takeaway:** We need RL approaches that exploit structure, respect constraints, and start from what we know.

---

# Methodoloy in the Litterature

- A recent review (**Khabbazi et al., 2025**) found **71% of field demonstrations** used **protocols prone to unreliable estimates**.
- Even among the credible **29%**, reported gains are modest: **13–16% energy or cost savings** that rarely account for deployment costs.
- Many studies assume perfect digital twins, unlimited simulation rollouts, and simplified objectives (e.g., only energy use).

  - **71 % of field demonstrations** of MPC and RL for HVAC used **protocols prone to unreliable estimates**.
  - Even among the credible **29 %**, reported gains are modest:

    - **13–16 % energy or cost savings**
    - Rarely account for deployment costs or operational complexity.


**Implication:** Naive RL approaches fail not only in practice but also in methodology.


---


# Onboarding Reality: 2 Days or 2 Years?
## Why Inductive Bias Matters

Onboarding cannot take **months**. Full digital twins are slow and expensive.  
- Pure black-box RL:  
  - Needs huge data  
  - Could take **years** to be reliable  
- Smarter approach:  
  - Add structure from physics and constraints  
  - Model energy flows, not just correlations  

**First-law residual:**  
$$
R(t) = (\text{heat in} + \text{heat out} + \text{work}) - \text{change in stored energy}
$$

**Key point:** Inductive bias accelerates learning and keeps policies feasible and safe.

---

# Physics Saves You Samples

<img src="thermodynamics.png" alt="Thermodynamics" style="height: 40%; width: auto; display: block; margin: 0 auto;">

- Energy **cannot disappear**. When you add heat to a zone, the net energy balance must rise.  
- Disturbances (e.g., an open window) simply **add an extra heat-loss term**; the balance still holds.  
- Pure black-box models can break this rule, predicting temperature drops while heating is applied.  


---

# Thermodynamic Structure in One Equation

For a single zone model, we have: 

$$
C \frac{dT}{dt} \;=\; \sum_{i} U_i \,(T_i - T)\;+\; Q_{\text{HVAC}}\;+\; Q_{\text{internal}}
$$

- $C$ thermal capacitance of the zone  
- $U_i$ conductance to neighboring zones or outside  
- $Q_{\text{HVAC}}$ heating / cooling input  
- $Q_{\text{internal}}$ gains from occupants and equipment  

Adding this single ODE pins the model to energy conservation, cuts the search space, and speeds learning.

**Moving to continuous-time models is natural here!**

---

# From First Law → Neural ODE

**How we enforce energy balance**

1. Split the latent state:  $z = \bigl[z_{\mathcal E}\;,\; z_{T}\bigr]$
   where $z_{\mathcal E}$ tracks **HVAC energy** and $z_{T}$ tracks **temperature**.

2. Use a structured vector field that **couples the two** exactly as the first-law ODE:  
   $$
   \begin{aligned}
   \dot z_{\mathcal E} &= P_\theta(s,a,d) \\
   \dot z_{T} &= \tfrac{1}{C_p}\,\bigl(Q_\theta(s,a,d) + \eta P_\theta(s,a,d)\bigr)
   \end{aligned}
   $$
   – $P_\theta$ and $Q_\theta$ are learned networks for HVAC power and other gains.  
   – $C_p$ and $\eta$ are learnable or fixed physical parameters.

3. Train end-to-end with the usual MSE loss; the structure **guarantees** energy-conserving dynamics, so no extra penalty terms are needed.
---

# Fast Deployment, Better Control

- **Physics-informed Neural ODE + gradient planning**
  - 92 % DR events satisfied (vs 86 % rule-based)  
  - 15 % less comfort drift during events  
- **Sample-efficient**  
  - Same control accuracy with **≈ 50 % less data than a discrete RNN** (Taboga et al., 2024)  
- **Field validation**  
  - Applied to real data from 5 commercial buildings in collaboration with **BrainboxAI (now Trane)**
  - Onboarding time measured in **days, not months**

**Takeaway:** encoding basic physics cuts data needs and hits DR targets sooner — a practical path to rapid HVAC deployments.

---
# RL for Supermarket Refrigeration

Let me tell you about this other industry project, this time with IVADO Labs and a major Canadian grocery chain

- Refrigeration accounts for **40–60% of store electricity use**
- Energy bills are about **\$4–5 per square foot annually**, which is more than \$200,000 for a typical store
- Grocery profit margins are very thin, typically **1–3% of sales**
- Every **\$1 saved on energy equals about \$50–60 in additional sales**

**Implication:**
Reducing refrigeration energy by just **1%** can have the same impact on profits as a large increase in revenue. This is why the problem matters for industry.

---

# Compressor Rack Optimization

- Refrigeration racks in the machine room supply dozens of display cases through a shared **suction header**
- **Suction pressure** controls the evaporating temperature in all cases
- **Floating suction pressure** = adjusting this setpoint over time

  - Lower pressure → colder coils, higher energy use
  - Higher pressure → energy savings, risk of food warming
- **Challenge**

  - One control affects the whole network
  - Loads change unpredictably (door openings, defrost, ambient)
  - Fast compressor cycling vs slow case temperature dynamics

**Goal:** Minimize energy use while maintaining food safety and operational reliability.

---

# What a Solution Could Have Been

- Detailed physics-based models exist for supermarket refrigeration

  - Example: Danfoss benchmark model (Larsen et al., 2007)
  - Multi-state ODEs for air, product, walls, and refrigerant dynamics
- These models can deliver accurate simulation and MPC **but**:

  - Require deep expertise and manual tuning
  - Depend on full visibility into equipment and sensors
  - Heavy integration: piping data through APIs and custom databases
- For a large, diverse fleet:

  - Hard to replicate and maintain across hundreds of stores
  - Slows onboarding by weeks or months

**Reality:** Full digital twin approaches are powerful but not practical for rapid, large-scale deployment.

---

# What Works in Practice

- There is no single RL module that solves everything
- Real-world deployments are **multi-pronged**:

  - Optimize setpoints when safe and feasible
  - Detect and correct **fault conditions** early: often the biggest savings

    - Stuck valves, failed sensors, clogged filters
    - Symptoms: compressors short-cycling, suction pressure oscillating, energy spikes ([Axiom Cloud, 2023](https://axiomcloud.ai/blog/floating-suction))
  - Alert technicians instead of pushing aggressive real-time control
- Why this matters:

  - A single fix can recover **10–20% energy** without changing any control policy
  - Safer and faster than blindly tuning suction setpoints at high frequency



---
# When We Optimize Live

- Focus shifts to **controller tuning** for energy savings; not black box policies. 

- What works:

  - **Simple thermodynamic model + BO-MPC**
  - Or **contextual BO online** for cold-start
- Benefits:

  - Sample-efficient
  - Safe, adaptive to ambient and load conditions
- Updates happen over **hours or days**, not every second

---

# What We Learned from These Projects

- Real-world deployment is not about picking PPO or SAC
- The hardest part is **modeling and integration**, not the algorithm
- One-size-fits-all RL does not exist

  - Practical solutions combine analytics, fault detection, and adaptive control
- Safe, sample-efficient methods are essential for fast onboarding

**Bottom line:** Progress in modeling will bring RL closer to reality.

---

# The Next Frontier: Transfer and Scale

- Every building today is a new project
- Transfer learning would change the game:

  - Learn from many buildings and adapt to new ones with little data
- Why this matters:

  - The HVAC market is massive; even small gains have big impact
- Why hard:

  - Multi-morphology systems, diverse sensors, incomplete logs
  - No internet-scale datasets yet
- First signs:

  - AutoBEM models over **125M building footprints** in the US
  - Similar initiatives in Canada and Quebec

This is the next ALE moment for energy systems.

---

# RL Beyond Policies: Modeling and Guidance

- LLMs can encode standards, safety rules, and expert practices
- Promising directions:

  - Guide data collection and model building
  - Use LLM reasoning in **Bayesian optimization** for tuning
  - Fault detection and anomaly diagnosis from telemetry
- Long-term opportunities:

  - Generate symbolic or hybrid models
  - Assist in defining agents via structured code or constraints

The goal: move from blind exploration to expert-informed strategies.

---

# What the Future Agent Looks Like

- Not a giant black-box policy
- A modular system:

  - Physics-informed models for dynamics
  - Transferable priors for different HVAC configurations
  - A reasoning layer for fault handling and planning
- Foundation models for control are the big opportunity

  - Pretraining across buildings
  - Fast adaptation to new sites

**Takeaway:** RL in the real world will look like a hybrid of learning, modeling, and reasoning.
