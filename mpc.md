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

# Model Predictive Control


The methods seen so far, whether in discrete or continuous-time, were deterministic trajectory optimization methods. Given an initial state, they provide a prescription of controls to apply as a function of time. In the collocation case, we would also obtain a state trajectory corresponding to those controls as a by-product. The control function, $\mathbf{u}[i]$ (in discrete time) or $u(t)$ (in continuous-time), is blind to the system's state at any point in time. It simply reads off the values stored in memory or performs some form of interpolation in time. This approach is optimal under the assumption that our system is deterministic. We know that no matter how many times we repeat the experiment from the same initial state, we will always get the same result; hence, reapplying the same controls in the same order is sufficient.

However, no matter how good our model is, real-life deployment of our controller will inevitably involve prediction errors. In such cases, a simple control function that only considers the current time—an open-loop controller—won't suffice. We must inform our controller that the real world is likely not in the state it thinks it's in: we need to provide feedback to close the loop. Depending on the structure of the noise affecting our system, we may encounter feedback controllers that depend on the entire history (in the case of partial observability) or just the current state (under perfect observability). Dynamic programming methods will provide us with solution methods to derive such controllers. These methods exist for both the continuous-time setting (via the Hamilton-Jacobi equations) and the discrete setting through the Bellman optimality equations. It also provides us with the necessary framework to address partially observable systems (for which we can't directly measure the state) using the POMDP framework. We will cover these solution methods in this chapter.

# Closing the Loop by Replanning

There exists a simple recipe for closing the loop: since we will likely end up in states that we haven't planned for, we might as well recompute our solution as frequently as possible using the updated state information as initial state to our trajectory optimization procedure. This replanning or reoptimization strategy is called Model Predictive Control (MPC). Given any trajectory optimization algorithm, we can turn it into a closed-loop variant using MPC.

Consider a trajectory optimization problem in Bolza form, 

$$
\begin{aligned}
\text{minimize} \quad & c(\mathbf{x}(t_0 + T)) + \int_{t_0}^{t_0 + T} c(\mathbf{x}(t), \mathbf{u}(t)) \, dt \\
\text{subject to} \quad & \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)) \\
& \mathbf{g}(\mathbf{x}(t), \mathbf{u}(t)) \leq \mathbf{0} \\
& \mathbf{u}_{\text{min}} \leq \mathbf{u}(t) \leq \mathbf{u}_{\text{max}} \\
\text{given} \quad & \mathbf{x}(t_0) = \mathbf{x}_{\text{current}} \enspace .
\end{aligned}
$$

MPC then proceeds as follows. At the current time $ t_0 $, the MPC controller considers a prediction horizon $ T $ over which it optimizes the future trajectory of the system. The controller then solves the trajectory optimization problem with the initial state set to the current state $ \mathbf{x}_{\text{current}} = \mathbf{x}(t_0) $. This yields an optimal control trajectory $ \mathbf{u}^*(t) $ for $ t \in [t_0, t_0 + T] $. 

However,  Instead of applying the entire computed control trajectory, the controller extracts only the first part of the solution, namely $ \mathbf{u}^*(t_0) $, and applies it to the system. This strategy is called *receding horizon control*. The idea is that the control $ \mathbf{u}^*(t_0) $ is based on the best prediction available at time $ t_0 $, considering the current state of the system and expected future disturbances.

After applying the first control input, the system evolves according to its dynamics, and the controller measures the new state at the next time step, $ t_0 + \Delta t $. Using this updated state as the new initial condition, the MPC controller re-solves the trajectory optimization problem over a shifted prediction horizon $ [t_0 + \Delta t, t_0 + \Delta t + T] $.

This procedure is then repeated ad infinitum or until the end of overall control problem. More concisely, here's a pseudo-code showing the general blueprint of MPC methods: 

````{prf:algorithm} Non-linear Model Predictive Control
:label: alg-mpc

**Input:**
- Prediction horizon $ T $
- Time step $ \Delta t $
- Initial state $ \mathbf{x}(t_0) $
- Cost functions $ c(\mathbf{x}(t), \mathbf{u}(t)) $ and $ c(\mathbf{x}(t_0 + T)) $
- System dynamics $ \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)) $
- Constraints $ \mathbf{g}(\mathbf{x}(t), \mathbf{u}(t)) \leq \mathbf{0} $
- Control limits $ \mathbf{u}_{\text{min}} \leq \mathbf{u}(t) \leq \mathbf{u}_{\text{max}} $

**Output:**
- Control input sequence $ \{\mathbf{u}(t)\} $ applied to the system

**Initialization:**
1. Set $ t \leftarrow t_0 $
2. Measure initial state $ \mathbf{x}_{\text{current}} \leftarrow \mathbf{x}(t) $

**Procedure:**

3. **Repeat** until the end of the control task:

   4. **Solve the following optimization problem:**

   $$
   \begin{aligned}
   \text{minimize} \quad & c(\mathbf{x}(t + T)) + \int_{t}^{t + T} c(\mathbf{x}(\tau), \mathbf{u}(\tau)) \, d\tau \\
   \text{subject to} \quad & \dot{\mathbf{x}}(\tau) = \mathbf{f}(\mathbf{x}(\tau), \mathbf{u}(\tau)) \quad \forall \tau \in [t, t + T] \\
   & \mathbf{g}(\mathbf{x}(\tau), \mathbf{u}(\tau)) \leq \mathbf{0} \quad \forall \tau \in [t, t + T] \\
   & \mathbf{u}_{\text{min}} \leq \mathbf{u}(\tau) \leq \mathbf{u}_{\text{max}} \quad \forall \tau \in [t, t + T] \\
   \text{given} \quad & \mathbf{x}(t) = \mathbf{x}_{\text{current}}
   \end{aligned}
   $$
   - Obtain the optimal control trajectory $ \mathbf{u}^*(\tau) $ and the optimal state trajectory $ \mathbf{x}^*(\tau) $ for $ \tau \in [t, t + T] $.

   5. **Apply the first control input:**

   $$
   \mathbf{u}(t) \leftarrow \mathbf{u}^*(t)
   $$
   - Apply $ \mathbf{u}(t) $ to the system.

   6. **Advance the system state:**
   - Let the system evolve under the control $ \mathbf{u}(t) $ according to the dynamics $ \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)) $.
   - Update $ t \leftarrow t + \Delta t $.

   7. **Measure the new state:**

   $$
   \mathbf{x}_{\text{current}} \leftarrow \mathbf{x}(t)
   $$

8. **End Repeat**

````

## Example: Propofol Infusion Control 

This problem explores the control of propofol infusion in total intravenous anesthesia (TIVA). Our presentation follows the problem formulation developped by {cite:t}`Sawaguchi2008`. The primary objective is to maintain the desired level of unconsciousness while minimizing adverse reactions and ensuring quick recovery after surgery. 

The level of unconsciousness is measured by the Bispectral Index (BIS), which is obtained using an electroencephalography (EEG) device. The BIS ranges from $0$ (complete suppression of brain activity) to $100$ (fully awake), with the target range for general anesthesia typically between $40$ and $60$.

The goal is to design a control system that regulates the infusion rate of propofol to maintain the BIS within the target range. This can be formulated as an optimal control problem:

$$
\begin{align*}
\min_{u(t)} & \int_{0}^{T} \left( BIS(t) - BIS_{\text{target}} \right)^2 + \gamma u(t)^2 \, dt \\
\text{subject to:} \\
\dot{x}_1 &= -(k_{10} + k_{12} + k_{13})x_1 + k_{21}x_2 + k_{31}x_3 + \frac{u(t)}{V_1} \\
\dot{x}_2 &= k_{12}x_1 - k_{21}x_2 \\
\dot{x}_3 &= k_{13}x_1 - k_{31}x_3 \\
\dot{x}_e &= k_{e0}(x_1 - x_e) \\
BIS(t) &= E_0 - E_{\text{max}}\frac{x_e^\gamma}{x_e^\gamma + EC_{50}^\gamma}
\end{align*}
$$

Where:
- $u(t)$ is the propofol infusion rate (mg/kg/h)
- $x_1$, $x_2$, and $x_3$ are the drug concentrations in different body compartments
- $x_e$ is the effect-site concentration
- $k_{ij}$ are rate constants for drug transfer between compartments
- $BIS(t)$ is the Bispectral Index
- $\gamma$ is a regularization parameter penalizing excessive drug use
- $E_0$, $E_{\text{max}}$, $EC_{50}$, and $\gamma$ are parameters of the pharmacodynamic model

The specific dynamics model used in this problem is so-called "Pharmacokinetic-Pharmacodynamic Model" and consists of three main components:

1. **Pharmacokinetic Model**, which describes how the drug distributes through the body over time. It's based on a three-compartment model:
   - Central compartment (blood and well-perfused organs)
   - Shallow peripheral compartment (muscle and other tissues)
   - Deep peripheral compartment (fat)

2. **Effect Site Model**, which represents the delay between drug concentration in the blood and its effect on the brain.

3. **Pharmacodynamic Model** that relates the effect-site concentration to the observed BIS.

The propofol infusion control problem presents several interesting challenges from a research perspective. 
First, there is a delay in how fast the drug can reach a different compartments in addition to the BIS measurements which can lag. This could lead to instability if not properly addressed in the control design. 

Furthermore, every patient is different from another. Hence, we cannot simply learn a single controller offline and hope that it will generalize to an entire patient population. We will account for this variability through Model Predictive Control (MPC) and dynamically adapt to the model mismatch through replanning. How a patient will react to a given dose of drug also varies and must be carefully controlled to avoid overdoses. This adds an additional layer of complexity since we have to incorporate safety constraints. Finally, the patient might suddenly change state, for example due to surgical stimuli, and the controller must be able to adapt quickly to compensate for the disturbance to the system.

```{code-cell} ipython3
:tags: [hide-input]
:load: code/hypnosis_control_nmpc.py
```


### Successive Linearization and Quadratic Approximations

For many regulation and tracking problems, the nonlinear dynamics and costs we encounter can be approximated locally by linear and quadratic functions. The basic idea is to linearize the system around the current operating point and approximate the cost with a quadratic form. This reduces each MPC subproblem to a **quadratic program (QP)**, which can be solved reliably and very quickly using standard solvers.

Suppose the true dynamics are nonlinear,

$$
\mathbf{x}_{k+1} = f(\mathbf{x}_k,\mathbf{u}_k).
$$

Around a nominal trajectory $(\bar{\mathbf{x}}_k,\bar{\mathbf{u}}_k)$, we take a first-order expansion:

$$
\mathbf{x}_{k+1} \approx f(\bar{\mathbf{x}}_k,\bar{\mathbf{u}}_k) 
+ A_k(\mathbf{x}_k - \bar{\mathbf{x}}_k) 
+ B_k(\mathbf{u}_k - \bar{\mathbf{u}}_k),
$$

with Jacobians

$$
A_k = \frac{\partial f}{\partial \mathbf{x}}(\bar{\mathbf{x}}_k,\bar{\mathbf{u}}_k), 
\qquad
B_k = \frac{\partial f}{\partial \mathbf{u}}(\bar{\mathbf{x}}_k,\bar{\mathbf{u}}_k).
$$

Similarly, if the stage cost is nonlinear,

$$
c(\mathbf{x}_k,\mathbf{u}_k),
$$

we approximate it quadratically near the nominal point:

$$
c(\mathbf{x}_k,\mathbf{u}_k) \;\approx\; 
\|\mathbf{x}_k - \mathbf{x}_k^{\text{ref}}\|_{Q_k}^2 
+ \|\mathbf{u}_k - \mathbf{u}_k^{\text{ref}}\|_{R_k}^2,
$$

with positive semidefinite weighting matrices $Q_k$ and $R_k$.

The resulting MPC subproblem has the form

$$
\begin{aligned}
\min_{\mathbf{x}_{0:N},\mathbf{u}_{0:N-1}} \quad &
\|\mathbf{x}_N - \mathbf{x}_N^{\text{ref}}\|_{P}^2
+ \sum_{k=0}^{N-1} 
\left(
\|\mathbf{x}_k - \mathbf{x}_k^{\text{ref}}\|_{Q_k}^2
+ \|\mathbf{u}_k - \mathbf{u}_k^{\text{ref}}\|_{R_k}^2
\right) \\
\text{s.t.} \quad &
\mathbf{x}_{k+1} = A_k \mathbf{x}_k + B_k \mathbf{u}_k + d_k, \\
& \mathbf{u}_{\min} \leq \mathbf{u}_k \leq \mathbf{u}_{\max}, \\
& \mathbf{x}_{\min} \leq \mathbf{x}_k \leq \mathbf{x}_{\max}, \\
& \mathbf{x}_0 = \mathbf{x}_{\text{current}} ,
\end{aligned}
$$

where $d_k = f(\bar{\mathbf{x}}_k,\bar{\mathbf{u}}_k) - A_k \bar{\mathbf{x}}_k - B_k \bar{\mathbf{u}}_k$ captures the local affine offset.

Because the dynamics are now linear and the cost quadratic, this optimization problem is a convex quadratic program. Quadratic programs are attractive in practice: they can be solved at kilohertz rates with mature numerical methods, making them the backbone of many real-time MPC implementations.

At each MPC step, the controller updates its linearization around the new operating point, constructs the local QP, and solves it. The process repeats, with the linear model and quadratic cost refreshed at every reoptimization. Despite the approximation, this yields a closed-loop controller that inherits the fast computation of QPs while retaining the ability to track trajectories of the underlying nonlinear system.

# The Landscape of MPC Variants

Once the basic idea of receding horizon control is clear, it is helpful to see how the same backbone accommodates many variations. In every case, we transcribe the continuous problem to a nonlinear program of the form

$$
\begin{aligned}
\text{minimize}\quad & c_T(\mathbf{x}_N)+\sum_{k=0}^{N-1} w_k\,c(\mathbf{x}_k,\mathbf{u}_k) \\
\text{subject to}\quad & \mathbf{x}_{k+1}=\mathbf{F}_k(\mathbf{x}_k,\mathbf{u}_k) \\
& \mathbf{g}(\mathbf{x}_k,\mathbf{u}_k)\leq \mathbf{0}, \\
& \mathbf{x}_{\min}\leq \mathbf{x}_k\leq \mathbf{x}_{\max},\quad \mathbf{u}_{\min}\leq \mathbf{u}_k\leq \mathbf{u}_{\max}, \\
& \mathbf{x}_0=\mathbf{x}_{\text{current}} ,
\end{aligned}
$$

with \$\mathbf{F}\_k\$ the chosen discretization of the dynamics and \$w\_k\$ the quadrature weights. From this skeleton, several families of MPC emerge.

In **tracking MPC**, the stage and terminal costs are quadratic penalties on deviation from a reference trajectory,

$$
c(\mathbf{x}_k,\mathbf{u}_k)=\|\mathbf{x}_k-\mathbf{x}_k^{\text{ref}}\|_{\mathbf{Q}}^2+\|\mathbf{u}_k-\mathbf{u}_k^{\text{ref}}\|_{\mathbf{R}}^2 , \qquad c_T(\mathbf{x}_N)=\|\mathbf{x}_N-\mathbf{x}_N^{\text{ref}}\|_{\mathbf{P}}^2 .
$$

This is the industrial workhorse, ensuring that the system follows a desired profile within limits.

In **regulatory MPC**, the reference is fixed at an equilibrium \$(\mathbf{x}^e,\mathbf{u}^e)\$, and the quadratic penalty encourages return to this point. Terminal constraints are often added so that stability can be guaranteed.

In **economic MPC**, the quadratic structure disappears altogether. Instead, the cost encodes economic performance,

$$
c(\mathbf{x}_k,\mathbf{u}_k)=c_{\text{econ}}(\mathbf{x}_k,\mathbf{u}_k),
$$

for instance energy cost, profit, or resource efficiency. The optimization then steers the system not toward a setpoint but toward economically optimal regimes.

When uncertainty is represented by bounded sets, one arrives at **robust MPC**, which seeks controls that satisfy the constraints for all admissible disturbances. The resulting NLP has a min–max structure. A tractable alternative is tube MPC, where the nominal optimization is carried out with tightened constraints to guarantee feasibility of the true system under a disturbance feedback law.

If uncertainty is stochastic, the formulation turns into **stochastic MPC**, where the objective is the expected cost and the constraints are imposed with high probability,

$$
\mathbb{P}\!\left[\mathbf{g}(\mathbf{x}_k,\mathbf{u}_k)\leq \mathbf{0}\right]\geq 1-\varepsilon .
$$

Scenario-based versions replace expectations by a sampled, deterministic problem.

Some systems require discrete choices, such as switching devices on or off. **Hybrid MPC** introduces integer variables into the transcription, producing a mixed-integer NLP that can handle such logic.

For large networks of subsystems, **distributed MPC** coordinates several local predictive controllers that optimize their own subsystems while communicating through coupling constraints.

Finally, in settings where models are uncertain or slowly drifting, one finds **adaptive or learning-based MPC**, which uses parameter estimation or machine learning to update the model \$\mathbf{F}\_k(\cdot;\theta\_t)\$ and possibly the cost function. The optimization step remains the same, but the model evolves as more data are collected.

These formulations illustrate that MPC is less a single algorithm than a recipe. The scaffolding is always the same: finite horizon prediction, state and input constraints, and receding horizon application of the control. What changes from one variant to another is the cost function, the way uncertainty is treated, the presence of discrete decisions, or the architecture across multiple agents.


## The Landscape of MPC Variants

Once the basic idea of receding-horizon control is clear, it is helpful to see how the same backbone accommodates many variations. In every case, we transcribe the continuous-time optimal control problem into a nonlinear program of the form

$$
\begin{aligned}
    \text{minimize} \quad & c(\mathbf{x}_N) + \sum_{k=0}^{N-1} w_k\,c(\mathbf{x}_k, \mathbf{u}_k) \\
    \text{subject to} \quad & \mathbf{x}_{k+1} = \mathbf{F}_k(\mathbf{x}_k, \mathbf{u}_k) \\
                            & \mathbf{g}(\mathbf{x}_k, \mathbf{u}_k) \leq \mathbf{0} \\
                            & \mathbf{x}_{\min} \leq \mathbf{x}_k \leq \mathbf{x}_{\max} \\
                            & \mathbf{u}_{\min} \leq \mathbf{u}_k \leq \mathbf{u}_{\max} \\
    \text{given} \quad & \mathbf{x}_0 = \hat{\mathbf{x}}(t) \enspace .
\end{aligned}
$$

The components in this NLP come from discretizing the continuous-time problem with a fixed horizon $[t, t+T]$ and step size $\Delta t$. The stage weights $w_k$ and discrete dynamics $\mathbf{F}_k$ are determined by the choice of quadrature and integration scheme. With this blueprint in place, the rest is a matter of interpretation: how we define the cost, how we handle uncertainty, how we treat constraints, and what structure we exploit.

### Tracking MPC

The most common setup is reference tracking. Here, we are given time-varying target trajectories $(\mathbf{x}_k^{\text{ref}}, \mathbf{u}_k^{\text{ref}})$, and the controller’s job is to keep the system close to these. The cost is typically quadratic:

$$
\begin{aligned}
    c(\mathbf{x}_k, \mathbf{u}_k) &= \| \mathbf{x}_k - \mathbf{x}_k^{\text{ref}} \|_{\mathbf{Q}}^2 + \| \mathbf{u}_k - \mathbf{u}_k^{\text{ref}} \|_{\mathbf{R}}^2 \\
    c(\mathbf{x}_N) &= \| \mathbf{x}_N - \mathbf{x}_N^{\text{ref}} \|_{\mathbf{P}}^2 \enspace .
\end{aligned}
$$

When dynamics are linear and constraints are polyhedral, this yields a convex quadratic program at each time step.

### Regulatory MPC

In regulation tasks, we aim to bring the system back to an equilibrium point $(\mathbf{x}^e, \mathbf{u}^e)$, typically in the presence of disturbances. This is simply tracking MPC with constant references:

$$
\begin{aligned}
    c(\mathbf{x}_k, \mathbf{u}_k) &= \| \mathbf{x}_k - \mathbf{x}^e \|_{\mathbf{Q}}^2 + \| \mathbf{u}_k - \mathbf{u}^e \|_{\mathbf{R}}^2 \\
    c(\mathbf{x}_N) &= \| \mathbf{x}_N - \mathbf{x}^e \|_{\mathbf{P}}^2 \enspace .
\end{aligned}
$$

To guarantee stability, it is common to include a terminal constraint $\mathbf{x}_N \in \mathcal{X}_f$, where $\mathcal{X}_f$ is a control-invariant set under a known feedback law.


### Economic MPC

Not all systems operate around a reference. Sometimes the goal is to optimize a true economic objective (eg. energy cost, revenue, efficiency) directly. This gives rise to **economic MPC**, where the cost functions reflect real operational performance:

$$
c(\mathbf{x}_k, \mathbf{u}_k) = c_{\text{op}}(\mathbf{x}_k, \mathbf{u}_k), \qquad
c(\mathbf{x}_N) = c_{\text{op},T}(\mathbf{x}_N) \enspace .
$$

There is no reference trajectory here. The optimal behavior emerges from the cost itself. In this setting, standard stability arguments no longer apply automatically, and one must be careful to add terminal penalties or constraints that ensure the closed-loop system remains well-behaved.

### Robust MPC

Some systems are exposed to external disturbances or small errors in the model. In those cases, we want the controller to make decisions that will still work no matter what happens, as long as the disturbances stay within some known bounds. This is the idea behind **robust MPC**.

Instead of planning a single trajectory, the controller plans a "nominal" path (what would happen in the absence of any disturbance) and then adds a feedback correction to react to whatever disturbances actually occur. This looks like:

$$
\mathbf{u}_k = \bar{\mathbf{u}}_k + \mathbf{K} (\mathbf{x}_k - \bar{\mathbf{x}}_k) \enspace ,
$$

where $\bar{\mathbf{u}}_k$ is the planned input and $\mathbf{K}$ is a feedback gain that pulls the system back toward the nominal path if it deviates.

Because we know the worst-case size of the disturbance, we can estimate how far the real state might drift from the plan, and "shrink" the constraints accordingly. The result is that the nominal plan is kept safely away from constraint boundaries, so even if the system gets pushed around, it stays inside limits. This is often called **tube MPC** because the true trajectory stays inside a tube around the nominal one.

The main benefit is that we can handle uncertainty without solving a complicated worst-case optimization at every time step. All the uncertainty is accounted for in the design of the feedback $\mathbf{K}$ and the tightened constraints.


### Stochastic MPC

If disturbances are random rather than adversarial, a natural goal is to optimize expected cost while enforcing constraints probabilistically. This gives rise to **stochastic MPC**, in which:

* The cost becomes an expectation:

  $$
  \mathbb{E} \left[ c(\mathbf{x}_N) + \sum_{k=0}^{N-1} w_k\, c(\mathbf{x}_k, \mathbf{u}_k) \right]
  $$
* Constraints are allowed to be violated with small probability:

  $$
  \mathbb{P}[\mathbf{g}(\mathbf{x}_k, \mathbf{u}_k) \leq \mathbf{0}] \geq 1 - \varepsilon
  $$

In practice, expectations are approximated using a finite set of disturbance scenarios drawn ahead of time. For each scenario, the system dynamics are simulated forward using the same control inputs $\mathbf{u}_k$, which are shared across all scenarios to respect non-anticipativity. The result is a single deterministic optimization problem with multiple parallel copies of the dynamics, one per sampled future. This retains the standard MPC structure, with only moderate growth in problem size.

Despite appearances, this is not dynamic programming. There is no value function or tree of all possible paths. There is only a finite set of futures chosen a priori, and optimized over directly. This scenario-based approach is common in energy systems such as hydro scheduling, where inflows are uncertain but sample trajectories can be generated from forecasts.

Risk constraints are typically enforced across all scenarios or encoded using risk measures like CVaR. For example, one might penalize violations that occur in the worst $(1 - \alpha)\%$ of samples, while still optimizing expected performance overall.

### Hybrid and Mixed-Integer MPC

When systems involve discrete switches — such as on/off valves, mode selection, or combinatorial logic — the MPC problem must include integer or binary variables. These show up in constraints like

$$
\boldsymbol{\delta}_k \in \{0,1\}^m, \qquad \mathbf{u}_k \in \mathcal{U}(\boldsymbol{\delta}_k)
$$

along with mode-dependent dynamics and costs. The resulting formulation is a **mixed-integer nonlinear program** (MINLP). The receding-horizon idea is the same, but each solve is more expensive due to the combinatorial nature of the decision space.

### Distributed and Decentralized MPC

Large-scale systems often consist of interacting subsystems. Distributed MPC decomposes the global NLP into smaller ones that run in parallel, with coordination constraints enforcing consistency across shared variables:

$$
\sum_{i} \mathbf{H}^i \mathbf{z}^i_k = \mathbf{0} \qquad \text{(coupling constraint)}
$$

Each subsystem solves a local problem over its own state and input variables, then exchanges information with neighbors. Coordination can be done via primal–dual methods, ADMM, or consensus schemes, but each local block looks like a standard MPC problem.


### Adaptive and Learning-Based MPC

In practice, we may not know the true model $\mathbf{F}_k$ or cost function $c$ precisely. In **adaptive MPC**, these are updated online from data:

$$
\mathbf{x}_{k+1} = \mathbf{F}_k(\mathbf{x}_k, \mathbf{u}_k; \boldsymbol{\theta}_t), \qquad
c(\mathbf{x}_k, \mathbf{u}_k) = c(\mathbf{x}_k, \mathbf{u}_k; \boldsymbol{\phi}_t)
$$

The parameters $\boldsymbol{\theta}_t$ and $\boldsymbol{\phi}_t$ are learned in real time. When combined with policy distillation, value approximation, or trajectory imitation, this leads to overlaps with reinforcement learning where the MPC solutions act as supervision for a reactive policy.


## Computational Efficiency via Parametric Programming

Real-time model predictive control places strict limits on computation. In applications such as adaptive optics, the controller must run at kilohertz rates. A sampling frequency of 1000 Hz allows only one millisecond per step to compute and apply a control input. This makes efficiency a first-class concern.

The structure of MPC lends itself naturally to optimization reuse. Each time step requires solving a problem with the same dynamics and constraints. Only the initial state, forecasts, or reference signals change. Instead of treating each instance as a new problem, we can frame MPC as a *parametric optimization problem* and focus on how the solution evolves with the parameter.

## General Framework: Parametric Optimization

We begin with a general optimization problem indexed by a parameter $\boldsymbol{\theta} \in \Theta \subset \mathbb{R}^p$:

$$
\begin{aligned}
\min_{\mathbf{x} \in \mathbb{R}^n} \quad & f(\mathbf{x}; \boldsymbol{\theta}) \\
\text{s.t.} \quad & \mathbf{g}(\mathbf{x}; \boldsymbol{\theta}) \le \mathbf{0}, \\
& \mathbf{h}(\mathbf{x}; \boldsymbol{\theta}) = \mathbf{0}.
\end{aligned}
\tag{P\(_\theta\)}
$$

For each value of $\boldsymbol{\theta}$, we obtain a concrete optimization problem. The goal is not just to solve it once, but to understand how the optimizer $\mathbf{x}^\star(\boldsymbol{\theta})$ and value function

$$
v(\boldsymbol{\theta}) := \inf\{\, f(\mathbf{x}; \boldsymbol{\theta}) : \mathbf{x} \text{ feasible at } \boldsymbol{\theta}\,\}
$$

depend on $\boldsymbol{\theta}$.

When the problem is smooth and regular, the Karush–Kuhn–Tucker (KKT) conditions characterize optimality:

$$
\begin{aligned}
\nabla_{\mathbf{x}} f(\mathbf{x}; \boldsymbol{\theta})
+ \nabla_{\mathbf{x}} \mathbf{g}(\mathbf{x}; \boldsymbol{\theta})^\top \boldsymbol{\lambda}
+ \nabla_{\mathbf{x}} \mathbf{h}(\mathbf{x}; \boldsymbol{\theta})^\top \boldsymbol{\nu} &= 0, \\
\mathbf{g}(\mathbf{x}; \boldsymbol{\theta}) \le 0, \quad
\boldsymbol{\lambda} \ge 0, \quad
\lambda_i g_i(\mathbf{x}; \boldsymbol{\theta}) &= 0, \\
\mathbf{h}(\mathbf{x}; \boldsymbol{\theta}) &= 0.
\end{aligned}
\tag{KKT}
$$

If the active set remains fixed over changes in $\boldsymbol{\theta}$, the implicit function theorem ensures that the mappings

$$
\boldsymbol{\theta} \mapsto \mathbf{x}^\star(\boldsymbol{\theta}), \quad
\boldsymbol{\theta} \mapsto \boldsymbol{\lambda}^\star(\boldsymbol{\theta}), \quad
\boldsymbol{\theta} \mapsto \boldsymbol{\nu}^\star(\boldsymbol{\theta})
$$

are differentiable.

In linear and quadratic programming, this structure becomes even more tractable. Consider a linear program with affine dependence on $\boldsymbol{\theta}$:

$$
\min_{\mathbf{x}} \ \mathbf{c}(\boldsymbol{\theta})^\top \mathbf{x}
\quad \text{s.t.} \quad \mathbf{A}(\boldsymbol{\theta})\mathbf{x} \le \mathbf{b}(\boldsymbol{\theta}).
$$

Each active set determines a basis and thus a region in $\Theta$ where the solution is affine in $\boldsymbol{\theta}$. The feasible parameter space is partitioned into polyhedral regions, each with its own affine law.

Similarly, in strictly convex quadratic programs

$$
\min_{\mathbf{x}} \ \tfrac{1}{2} \mathbf{x}^\top \mathbf{H} \mathbf{x} + \mathbf{q}(\boldsymbol{\theta})^\top \mathbf{x}
\quad \text{s.t.} \quad \mathbf{A}\mathbf{x} \le \mathbf{b}(\boldsymbol{\theta}), \qquad \mathbf{H} \succ 0,
$$

each active set again leads to an affine optimizer, with piecewise-affine global structure and a piecewise-quadratic value function.

Parametric programming focuses on the structure of the map $\boldsymbol{\theta} \mapsto \mathbf{x}^\star(\boldsymbol{\theta})$, and the regions over which this map takes a simple form.


Great spot to zoom in. Here’s a tight, pedagogical add-on you can drop right after your KKT display. It states the implicit function theorem (IFT) in the minimal form you need, then applies it to the KKT system and gives the sensitivity formula you’ll use later.

### A quick reminder: the implicit function theorem

We often meet equations of the form

$$
F(y,\boldsymbol{\theta})=0,
$$

where $y\in\mathbb{R}^m$ are unknowns and $\boldsymbol{\theta}\in\mathbb{R}^p$ are parameters. The **implicit function theorem** says that, if $F$ is smooth and the Jacobian with respect to $y$,

$$
\frac{\partial F}{\partial y}(y^\star,\boldsymbol{\theta}^\star),
$$

is invertible at a solution $(y^\star,\boldsymbol{\theta}^\star)$, then in a neighborhood of $\boldsymbol{\theta}^\star$ there exists a unique smooth mapping $y(\boldsymbol{\theta})$ with $F(y(\boldsymbol{\theta}),\boldsymbol{\theta})=0$ and $y(\boldsymbol{\theta}^\star)=y^\star$. Moreover, its derivative is

$$
\frac{d y}{d\boldsymbol{\theta}}(\boldsymbol{\theta}^\star)
\;=\;
-\Big(\tfrac{\partial F}{\partial y}(y^\star,\boldsymbol{\theta}^\star)\Big)^{-1}
\;\tfrac{\partial F}{\partial \boldsymbol{\theta}}(y^\star,\boldsymbol{\theta}^\star).
$$

In words: if the square Jacobian in $y$ is nonsingular, the solution varies smoothly with the parameter, and we can differentiate it by solving one linear system.

Return to $(P_{\theta})$ and its KKT system. Collect the primal and dual variables into

$$
y \;:=\; (\mathbf{x},\,\boldsymbol{\lambda},\,\boldsymbol{\nu}),
$$

and write the KKT equations as a single residual

$$
F(y,\boldsymbol{\theta}) \;=\; 
\begin{bmatrix}
\nabla_{\mathbf{x}} f(\mathbf{x};\boldsymbol{\theta})
+ \nabla_{\mathbf{x}} \mathbf{g}(\mathbf{x};\boldsymbol{\theta})^\top \boldsymbol{\lambda}
+ \nabla_{\mathbf{x}} \mathbf{h}(\mathbf{x};\boldsymbol{\theta})^\top \boldsymbol{\nu} \\
\mathbf{h}(\mathbf{x};\boldsymbol{\theta}) \\
\mathbf{g}_\mathcal{A}(\mathbf{x};\boldsymbol{\theta})
\end{bmatrix}
\;=\; \mathbf{0}.
$$

Here $\mathcal{A}$ denotes the set of inequality constraints active at the solution (the complementarity part is encoded by keeping $\mathcal{A}$ fixed; see below).

To invoke IFT, we need the Jacobian $\partial F/\partial y$ to be invertible at $(y^\star,\boldsymbol{\theta}^\star)$. Standard regularity conditions that ensure this are:

* **LICQ (Linear Independence Constraint Qualification)** at $(\mathbf{x}^\star,\boldsymbol{\theta}^\star)$: the gradients of all active constraints are linearly independent.
* **Second-order sufficiency** on the critical cone (the Lagrangian Hessian is positive definite on feasible directions).
* **Strict complementarity** (optional but convenient): each active inequality has strictly positive multiplier.

Under these, the **KKT matrix**,

$$
K \;=\;
\frac{\partial F}{\partial y}(y^\star,\boldsymbol{\theta}^\star)
\;=\;
\begin{bmatrix}
\nabla^2_{\mathbf{x}\mathbf{x}} \mathcal{L}(\mathbf{x}^\star,\boldsymbol{\lambda}^\star,\boldsymbol{\nu}^\star;\boldsymbol{\theta}^\star)
& \nabla_{\mathbf{x}} \mathbf{g}_\mathcal{A}(\mathbf{x}^\star;\boldsymbol{\theta}^\star)^\top
& \nabla_{\mathbf{x}} \mathbf{h}(\mathbf{x}^\star;\boldsymbol{\theta}^\star)^\top \\
\nabla_{\mathbf{x}} \mathbf{g}_\mathcal{A}(\mathbf{x}^\star;\boldsymbol{\theta}^\star) & 0 & 0 \\
\nabla_{\mathbf{x}} \mathbf{h}(\mathbf{x}^\star;\boldsymbol{\theta}^\star) & 0 & 0
\end{bmatrix},
$$

is nonsingular. Here $\mathcal{L}=f+\boldsymbol{\lambda}^\top \mathbf{g}+\boldsymbol{\nu}^\top \mathbf{h}$.

The right-hand side sensitivity to parameters is

$$
G \;=\; \frac{\partial F}{\partial \boldsymbol{\theta}}(y^\star,\boldsymbol{\theta}^\star)
\;=\;
\begin{bmatrix}
\nabla_{\boldsymbol{\theta}}\nabla_{\mathbf{x}} f
+ \sum_{i\in\mathcal{A}} \lambda_i^\star \nabla_{\boldsymbol{\theta}}\nabla_{\mathbf{x}} g_i
+ \sum_j \nu_j^\star \nabla_{\boldsymbol{\theta}}\nabla_{\mathbf{x}} h_j \\
\nabla_{\boldsymbol{\theta}} \mathbf{h} \\
\nabla_{\boldsymbol{\theta}} \mathbf{g}_\mathcal{A}
\end{bmatrix}_{(\mathbf{x}^\star,\boldsymbol{\theta}^\star)} .
$$

IFT then gives **local differentiability of the optimizer and multipliers**:

$$
\frac{d y^\star}{d\boldsymbol{\theta}}(\boldsymbol{\theta}^\star)
\;=\; -\,K^{-1} G.
$$

The formula above is valid **as long as the active set $\mathcal{A}$ does not change**. If a constraint switches between active/inactive, the mapping remains piecewise smooth, but the derivative may jump. In MPC, this is exactly why warm-starts are very effective most of the time and occasionally require a refactorization when the active set flips.

In parametric MPC, $\boldsymbol{\theta}$ gathers the current state, references, and forecasts. The IFT tells us that, under regularity and a stable active set, the optimal trajectory and first input vary smoothly with $\boldsymbol{\theta}$. The linear map $-K^{-1}G$ is exactly the object used in sensitivity-based warm starts and real-time iterations: small changes in $\boldsymbol{\theta}$ can be propagated through a single KKT solve to update the primal–dual guess before taking one or two Newton/SQP steps.

### Predictor-Corrector MPC

We start with a smooth root-finding problem

$$
F(z)=0,\qquad F:\mathbb{R}^m\to\mathbb{R}^m.
$$

**Newton’s method** iterates

$$
z^{(t+1)} \;=\; z^{(t)} - \big[\nabla F(z^{(t)})\big]^{-1} F\big(z^{(t)}\big),
$$

or equivalently solves the linearized system

$$
\nabla F(z^{(t)})\,\Delta z^{(t)} = -F\big(z^{(t)}\big),\qquad z^{(t+1)}=z^{(t)}+\Delta z^{(t)}.
$$

Convergence is local and fast when the Jacobian is nonsingular and the initial guess is close.

Now suppose the root depends on a parameter:

$$
F\big(z,\theta\big)=0,\qquad \theta\in\mathbb{R}.
$$

We want the solution path $\theta\mapsto z^\star(\theta)$. **Numerical continuation** advances $\theta$ in small steps and uses the previous solution as a warm start for the next Newton solve. This is the simplest and most effective way to “track” solutions of parametric systems.

At a known solution $(z^\star,\theta^\star)$, differentiate $F(z^\star(\theta),\theta)=0$ with respect to $\theta$:

$$
\nabla_z F(z^\star,\theta^\star)\,\frac{dz^\star}{d\theta}(\theta^\star) \;+\; \nabla_\theta F(z^\star,\theta^\star) \;=\; 0.
$$

If $\nabla_z F$ is invertible (IFT conditions), the **tangent** is

$$
\frac{dz^\star}{d\theta}(\theta^\star) \;=\; -\big[\nabla_z F(z^\star,\theta^\star)\big]^{-1}\,\nabla_\theta F(z^\star,\theta^\star).
$$

This is exactly the **implicit differentiation** formula. Continuation uses it as a **predictor**:

$$
z_{\text{pred}} \;=\; z^\star(\theta^\star) \;+\; \Delta\theta\;\frac{dz^\star}{d\theta}(\theta^\star).
$$

Then a few **corrector** steps apply Newton to $F(\,\cdot\,,\theta^\star+\Delta\theta)=0$ starting from $z_{\text{pred}}$. If Newton converges quickly, the step $\Delta\theta$ was appropriate; otherwise reduce $\Delta\theta$ and retry.

For parametric KKT systems, set $y=(x,\lambda,\nu)$ and $F(y,\theta)=0$ the KKT residual with $\theta$ collecting state, references, forecasts. The **KKT matrix** $K=\partial F/\partial y$ and **parameter sensitivity** $G=\partial F/\partial \theta$ give the tangent

$$
\frac{dy^\star}{d\theta} \;=\; -\,K^{-1}G.
$$

Continuation then becomes:

1. **Predictor**: $y_{\text{pred}} = y^\star + (\Delta\theta)\,(-K^{-1}G)$.
2. **Corrector**: a few Newton/SQP steps on the KKT equations at the new $\theta$.

In MPC, this yields efficient **warm starts** across time: as the parameter $\theta_t$ (current state, references) changes slightly, we predict the new primal–dual point and correct with 1–2 iterations—often enough to hit tolerance in real time.


## Application to MPC

We now specialize this idea to the structure of finite-horizon MPC. Fix a prediction horizon $N$. At each time step, we solve a problem with fixed structure and varying data. Define

$$
\boldsymbol{\theta} := (\mathbf{x}_0,\, \mathbf{r},\, \mathbf{w}),
$$

which includes the current state $\mathbf{x}_0$, reference signals $\mathbf{r}$, and exogenous forecasts $\mathbf{w}$.

The finite-horizon problem becomes

$$
\begin{aligned}
\min_{z} \quad & J(z;\boldsymbol{\theta}) \\
\text{s.t.} \quad & c(z;\boldsymbol{\theta}) = 0 \\
& d(z;\boldsymbol{\theta}) \leq 0,
\end{aligned}
\tag{P\(_\theta\)}
$$

with decision variable $z = (\mathbf{x}_{0:N}, \mathbf{u}_{0:N-1})$. The equality constraints enforce dynamics and terminal conditions. The inequalities encode input and state bounds.

Solving $(P_\theta)$ produces an optimal trajectory $z^\star(\boldsymbol{\theta})$. The control law is the first input:

$$
\pi(\boldsymbol{\theta}) := \mathbf{u}_0^\star(\boldsymbol{\theta}).
$$

This mapping from parameter to input defines the MPC policy. Parametric programming helps us understand and exploit its structure to speed up evaluation.

## Two Approaches to Efficient MPC

Parametric structure can be used in two main ways: either to construct an explicit control law offline, or to warm-start the optimizer online using sensitivity information.

### Explicit MPC

When the problem is a linear or quadratic program with affine dependence on $\boldsymbol{\theta}$, we can work out the solution symbolically. The parameter space is partitioned into regions $\mathcal{R}_1, \dots, \mathcal{R}_M$, each associated with a fixed active set. On each region:

$$
z^\star(\boldsymbol{\theta}) = A_r\,\boldsymbol{\theta} + b_r,
\qquad
\pi(\boldsymbol{\theta}) = K_r\,\boldsymbol{\theta} + k_r.
$$

At runtime, we identify which region contains $\boldsymbol{\theta}$, then apply the corresponding affine formula. This approach avoids optimization entirely during deployment.

It requires storing the region definitions and control laws. The number of regions grows with horizon length and constraint count, which limits this approach to systems with low state dimension and short horizons.

### Sensitivity-Based MPC

When symbolic enumeration is intractable, we can still track how the solution varies locally. Suppose we have a solution $\bar{y} = (\bar{z}, \bar{\lambda}, \bar{\nu})$ at parameter $\bar{\boldsymbol{\theta}}$. The KKT system reads:

$$
F(y; \boldsymbol{\theta}) = 0.
$$

Differentiating with respect to $\boldsymbol{\theta}$,

$$
\frac{\partial F}{\partial y}(\bar{y}; \bar{\boldsymbol{\theta}})\, \mathrm{d}y
= - \frac{\partial F}{\partial \boldsymbol{\theta}}(\bar{y}; \bar{\boldsymbol{\theta}})\, \mathrm{d}\boldsymbol{\theta}.
$$

Let $K$ be the KKT matrix and $G$ the sensitivity of the residual. Then the sensitivity operator $T$ satisfies

$$
K T = -G \quad \Rightarrow \quad \mathrm{d}y = T\, \mathrm{d}\boldsymbol{\theta}.
$$

If $\boldsymbol{\theta}$ changes slightly, we update the primal-dual pair:

$$
y^{(0)} \leftarrow \bar{y} + T\,\Delta\boldsymbol{\theta},
$$

and use it as the starting point for Newton or SQP.

This is the basis of real-time iteration schemes. When the active set is stable, the warm start is accurate to first order. When it changes, we refactorize and repeat, still with far less effort than solving from scratch.


## Amortized Optimization and Neural Approximation of Controllers

The idea of reusing structure across similar optimization problems is not exclusive to parametric programming. In machine learning, a related concept known as **amortized optimization** aims to reduce the cost of repeated inference by replacing explicit optimization with a function that has been *learned* to approximate the solution map. This approach shifts the computational burden from online solving to offline training.

The goal is to construct a function $\hat{\pi}_{\phi}(\boldsymbol{\theta})$, typically parameterized by a neural network, that maps the input $\boldsymbol{\theta}$ to an approximate solution $\hat{z}^\star(\boldsymbol{\theta})$ or control action $\hat{\mathbf{u}}_0^\star(\boldsymbol{\theta})$. Once trained, this map can be evaluated quickly at runtime, with no need to solve an optimization problem explicitly.

Amortized optimization has emerged in several contexts:

* In **probabilistic inference**, where variational autoencoders (VAEs) amortize the computation of posterior distributions across a dataset.
* In **meta-learning**, where the objective is to learn a model that generalizes across tasks by internalizing how to adapt.
* In **hyperparameter optimization**, where learning a surrogate model can guide the search over configuration space efficiently.

This perspective has also begun to influence control. Recent work investigates how to **amortize nonlinear MPC (NMPC)** policies into neural networks. The training data come from solving many instances of the underlying optimal control problem offline. The resulting neural policy $\hat{\pi}_\phi$ acts as a differentiable, low-latency controller that can generalize to new situations within the training distribution.

Compared to explicit MPC, which partitions the parameter space and stores exact solutions region by region, amortized control smooths over the domain by learning an approximate policy globally. It is less precise, but scalable to high-dimensional problems where enumeration of regions is impossible.

Neural network amortization is advantageous due to the expressivity of these models. However, the challenge is ensuring **constraint satisfaction and safety**, which are hard to guarantee with unconstrained neural approximators. Hybrid approaches attempt to address this by combining a neural warm-start policy with a final projection step, or by embedding the network within a constrained optimization layer. Other strategies include learning structured architectures that respect known physics or control symmetries.


## Imitation Learning Framework
Consider a fixed horizon $N$ and parameter vector $\boldsymbol{\theta}$ encoding the current state, references, and forecasts. The oracle MPC controller solves

$$
\begin{aligned}
z^\star(\boldsymbol{\theta}) \in \arg\min_{z=(x_{0:N},u_{0:N-1})}
&\; J(z;\boldsymbol{\theta})\\
\text{s.t. }& x_{k+1}=f(x_k,u_k;\boldsymbol{\theta}),\quad k=0..N-1,\\
& g(x_k,u_k;\boldsymbol{\theta})\le 0,\; h(x_N;\boldsymbol{\theta})=0.
\end{aligned}
$$

The applied action is $\pi^\star(\boldsymbol{\theta}) := u_0^\star(\boldsymbol{\theta})$. Our goal is to learn a fast surrogate mapping $\hat{\pi}_\phi:\boldsymbol{\theta}\mapsto \hat u_0 \approx \pi^\star(\boldsymbol{\theta})$ that can be evaluated in microseconds, optionally followed by a safety projection layer.

**Supervised learning from oracle solutions.**
One first samples parameters $\boldsymbol{\theta}^{(i)}$ from the operational domain and solves the corresponding NMPC problems offline. The resulting dataset

$$
\mathcal{D} = \{ (\boldsymbol{\theta}^{(i)},\, u_0^\star(\boldsymbol{\theta}^{(i)})) \}_{i=1}^M
$$

is then used to train a neural network $\hat{\pi}_\phi$ by minimizing

$$
\min_\phi \; \frac{1}{M}\sum_{i=1}^M \big\|\hat{\pi}_\phi(\boldsymbol{\theta}^{(i)}) - u_0^\star(\boldsymbol{\theta}^{(i)})\big\|^2 .
$$

Once trained, the network acts as a surrogate for the optimizer, providing instantaneous evaluations that approximate the MPC law.



### Deployment Patterns

There are several ways to use an amortized controller once it has been trained. The simplest option is **direct amortization**, where the control input is taken to be $u = \hat{\pi}_\phi(\boldsymbol{\theta})$. In this case, the neural network provides the control action directly, with no optimization performed during deployment.

A second option is **amortization with projection**, where the network output $\tilde u = \hat{\pi}_\phi(\boldsymbol{\theta})$ is passed through a small optimization step, such as a quadratic program or barrier-function filter, in order to enforce constraints. This adds a negligible computational overhead but restores guarantees of feasibility and safety.

We could for example integrate a convex approximation of the MPC subproblem directly as a differentiable layer inside the network. The network proposes a candidate action $\tilde u$, which is then corrected through a small quadratic program:

$$
u = \arg\min_v \tfrac12\|v-\tilde u\|^2 \quad \text{s.t. } g(x,v)\le 0.
$$

Gradients are propagated through this correction using implicit differentiation, allowing the network to be trained end-to-end while retaining constraint satisfaction. This hybrid keeps the fast evaluation of a learned map while preserving the structure of MPC.

A third option is **amortized warm-starting**, where the neural network provides an initialization for one or two Newton or SQP iterations of the underlying NMPC problem. In this setting, the learned map delivers an excellent starting point, so the optimizer converges quickly and the cost of re-solving at each time step is greatly reduced.