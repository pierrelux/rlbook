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

# Dynamic Programming

# Open-Loop vs Closed-Loop Control 

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
\min_{u(t)} & \int_{0}^{T} \left( BIS(t) - BIS_{\text{target}} \right)^2 + \lambda u(t)^2 \, dt \\
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
- $\lambda$ is a regularization parameter penalizing excessive drug use
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

## Computational Efficiency Improvements  

One challenge of Model Predictive Control (MPC) is its computational cost. In real-time applications, such as adaptive optics, the controller may need to operate at extremely high frequencies—for example, 1000 Hz. In this scenario, the solver has just 1 millisecond to compute an optimal solution, pushing the limits of computational efficiency.

### Explicit MPC 

A potential solution to this problem is to offload some of the computational effort offline. Instead of solving the optimization problem at every time step during execution, we could attempt to **precompute solutions** for all potential states in advance. At first glance, this seems impractical—without leveraging specific structure or partitioning the state space intelligently, precomputing solutions for every possible state would be infeasible. However, with the right techniques, this approach becomes viable.

This is the essence of **explicit MPC**, which hinges on a subfield of mathematical programming known as multi-parametric (or simply parametric) programming.  A multiparametric programming problem can be described by the following formulation:

$$
\begin{array}{cl}
z(\boldsymbol{\theta}) = \min_{\mathbf{x}} & f(\mathbf{x}, \boldsymbol{\theta}) \\
\text { s.t. } & \mathbf{g}(\mathbf{x}, \boldsymbol{\theta}) \leq 0 \\
& \mathbf{h}(\mathbf{x}, \boldsymbol{\theta}) = 0 \\
& \mathbf{x} \in \mathbb{R}^n \\
& \boldsymbol{\theta} \in \mathbb{R}^m
\end{array}
$$

where:
- $\mathbf{x} \in \mathbb{R}^n$ are the decision variables,
- $\boldsymbol{\theta} \in \mathbb{R}^m$ are the parameters,
- $f(\mathbf{x}, \boldsymbol{\theta})$ is the objective function,
- $\mathbf{g}(\mathbf{x}, \boldsymbol{\theta}) \leq 0$ and $\mathbf{h}(\mathbf{x}, \boldsymbol{\theta}) = 0$ are the inequality and equality constraints, respectively. 

Parametric programming methods provides ways for efficiently evaluating $z(\boldsymbol{\theta})$ -- the **value function** -- by leveraging the structure of the solution space. In particular, it leverages the idea that the solution space can be partitioned into **critical regions**—regions of the parameter space where the optimal solution structure remains unchanged. Within each region, the solution can often be expressed as a **piecewise affine function** of the parameters, which is easy to represent and compute offline. 

In trajectory optimization problems, the initial state $\boldsymbol{x}_0$ can also be treated as a **parameter**. This transforms the problem into a parametric optimization problem, where $\boldsymbol{x}_0$ defines a family of optimization problems, each yielding a different optimal solution. The relationship between the parameters and solutions can be described using two key mappings:
- $\boldsymbol{u}^\star(\boldsymbol{x}_0)$: The optimal control sequence as a function of the initial state.
- $v(\boldsymbol{x}_0)$: The value function, which gives the optimal objective value for a given $\boldsymbol{x}_0$.

It is therefore at this level that parametric programming methods can come into play and provide efficient methods for computing the value function offline: that is without resorting to direct calls to a nonlinear programming solver for every new $\boldsymbol{x}$ encountered along a trajectory. 

#### Amortized Optimization and Neural Networks

The idea of solving an entire family of optimization problems efficiently is not unique to parametric programming. In machine learning, **amortized optimization** (or **amortized inference**) aims to "learn to optimize" by constructing models that generalize over a family of optimization problems. This approach is particularly relevant in applications such as hyperparameter optimization, meta-learning, and probabilistic inference.

In contrast to explicit MPC, which partitions the state space, amortized optimization typically uses **neural networks** to approximate the mapping from parameters to optimal solutions. This has led to recent explorations of **amortizing NMPC (Nonlinear MPC) controllers** into neural networks, blending the structure of MPC with the generalization power of neural networks. This represents a promising direction for creating efficient controllers that combine physics-based models, safety constraints, and the flexibility of learned models.


<!-- 
As usual, the KKT conditions provide necessary conditions for optimality:

1. **Stationarity**:  

   $$
   \nabla_{\mathbf{x}} f(\mathbf{x}^*, \boldsymbol{\theta}) + \sum_{i=1}^{p} \lambda_i^* \nabla_{\mathbf{x}} g_i(\mathbf{x}^*, \boldsymbol{\theta}) + \sum_{j=1}^{q} \nu_j^* \nabla_{\mathbf{x}} h_j(\mathbf{x}^*, \boldsymbol{\theta}) = 0
   $$
   where $\boldsymbol{\lambda}^* = (\lambda_1^*, \ldots, \lambda_p^*)$ are the Lagrange multipliers for the inequality constraints, and $\boldsymbol{\nu}^* = (\nu_1^*, \ldots, \nu_q^*)$ are the Lagrange multipliers for the equality constraints.

2. **Primal Feasibility**:  

   $$
   \mathbf{g}(\mathbf{x}^*, \boldsymbol{\theta}) \leq 0, \quad \mathbf{h}(\mathbf{x}^*, \boldsymbol{\theta}) = 0
   $$

3. **Dual Feasibility**:  

   $$
   \lambda_i^* \geq 0, \quad \forall i
   $$

4. **Complementary Slackness**:  

   $$
   \lambda_i^* g_i(\mathbf{x}^*, \boldsymbol{\theta}) = 0, \quad \forall i
   $$


We can combine the KKT conditions into a single system of equations, denoted as $\mathbf{F}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}, \boldsymbol{\theta}) = \mathbf{0}$, where:

$$
\mathbf{F}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}, \boldsymbol{\theta}) = \begin{pmatrix}
\nabla_{\mathbf{x}} f(\mathbf{x}, \boldsymbol{\theta}) + \sum_{i=1}^{p} \lambda_i \nabla_{\mathbf{x}} g_i(\mathbf{x}, \boldsymbol{\theta}) + \sum_{j=1}^{q} \nu_j \nabla_{\mathbf{x}} h_j(\mathbf{x}, \boldsymbol{\theta}) \\
\mathbf{g}(\mathbf{x}, \boldsymbol{\theta}) \\
\mathbf{h}(\mathbf{x}, \boldsymbol{\theta}) \\
\boldsymbol{\lambda} \odot \mathbf{g}(\mathbf{x}, \boldsymbol{\theta}) \\
\min(\boldsymbol{\lambda}, \mathbf{0})
\end{pmatrix} = \mathbf{0}
$$

Here, $\mathbf{F}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}, \boldsymbol{\theta}) = \mathbf{0}$ encapsulates the stationarity, primal feasibility, dual feasibility, and complementary slackness conditions. The symbol $\odot$ represents the element-wise product for the complementary slackness condition. -->

### Warmstarting and Predictor-Corrector MPC 

Another way in which we can speed up NMPC is by providing good initial guesses for the solver. When solving a series of optimization problems along a trajectory, it is likely that the solution to previous problem might be close to that of the current one. Hence, as a heuristic it often makes sense to "warmstart" from the previous solution. 

Another alternative is to extrapolate what the next solution ought to be based on the previous one. What we mean here is that rather than simply using the last solution as a guess for that of the current problem, we leverage the "sensitivity information" around the last solution to make a guess about where we might be going. This idea is reminiscent of predictor-corrector schemes which we have briefly discussed in the first chapter. 

To implement a predictor corrector MPC scheme, we need to understand how the optimal solution $\mathbf{x}^*(\boldsymbol{\theta})$ and the value function $z(\boldsymbol{\theta})$ change as the parameters $\boldsymbol{\theta}$ vary. We achieve this by applying the **implicit function theorem** to the **KKT (Karush-Kuhn-Tucker) conditions** of the parametric problem. The KKT conditions for a parametric optimization problem are necessary for optimality and can be written as:

$$
\mathbf{F}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}, \boldsymbol{\theta}) = 0,
$$

where $\mathbf{F}$ encapsulates the stationarity, primal feasibility, dual feasibility, and complementary slackness conditions from the KKT theorem. By treating $\mathbf{x}$, $\boldsymbol{\lambda}$, and $\boldsymbol{\nu}$ as functions of the parameters $\boldsymbol{\theta}$, the implicit function theorem guarantees that, under certain regularity conditions, these optimal variables are **continuously differentiable** with respect to $\boldsymbol{\theta}$. This allows us to compute **sensitivity derivatives**, which describe how small changes in $\boldsymbol{\theta}$ affect the optimal solution.

By leveraging this sensitivity information, we can predict changes in the optimal solution and "warm-start" the optimization process at the next time step in MPC. This concept is related to **numerical continuation**, where a complex optimization problem is solved by gradually transforming a simpler, well-understood problem into the more difficult one.

# Principle of Optimality

Unlike the methods we've discussed so far, dynamic programming takes a step back and considers not just a single optimization problem, but an entire family of related problems. This approach, while seemingly more complex at first glance, can often lead to efficient solutions.

Dynamic programming leverage the solution structure underlying many control problems that allows for a decomposition it into smaller, more manageable subproblems. Each subproblem is itself an optimization problem, embedded within the larger whole. This recursive structure is the foundation upon which dynamic programming constructs its solutions.

To ground our discussion, let us return to the domain of discrete-time optimal control problems (DOCPs). These problems frequently arise from the discretization of continuous-time optimal control problems. While the focus here will be on deterministic problems, it is worth noting that these concepts extend naturally to stochastic problems by taking the expectation over the random quantities.

Consider a typical DOCP of Bolza type:

$$
\begin{align*}
\text{minimize} \quad & J \triangleq c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t, \mathbf{u}_t) \\
\text{subject to} \quad 
& \mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t), \quad t = 1, \ldots, T-1, \\
& \mathbf{u}_{lb} \leq \mathbf{u}_t \leq \mathbf{u}_{ub}, \quad t = 1, \ldots, T, \\
& \mathbf{x}_{lb} \leq \mathbf{x}_t \leq \mathbf{x}_{ub}, \quad t = 1, \ldots, T, \\
\text{given} \quad & \mathbf{x}_1
\end{align*}
$$

Rather than considering only the total cost from the initial time to the final time, dynamic programming introduces the concept of cost from an arbitrary point in time to the end. This leads to the definition of the "cost-to-go" or "value function" $J_k(\mathbf{x}_k)$:

$$
J_k(\mathbf{x}_k) \triangleq c_T(\mathbf{x}_T) + \sum_{t=k}^{T-1} c_t(\mathbf{x}_t, \mathbf{u}_t)
$$

This function represents the total cost incurred from stage $k$ onwards to the end of the time horizon, given that the system is initialized in state $\mathbf{x}_k$ at stage $k$. Suppose the problem has been solved from stage $k+1$ to the end, yielding the optimal cost-to-go $J_{k+1}^\star(\mathbf{x}_{k+1})$ for any state $\mathbf{x}_{k+1}$ at stage $k+1$. The question then becomes: how does this information inform the decision at stage $k$?

Given knowledge of the optimal behavior from $k+1$ onwards, the task reduces to determining the optimal action $\mathbf{u}_k$ at stage $k$. This control should minimize the sum of the immediate cost $c_k(\mathbf{x}_k, \mathbf{u}_k)$ and the optimal future cost $J_{k+1}^\star(\mathbf{x}_{k+1})$, where $\mathbf{x}_{k+1}$ is the resulting state after applying action $\mathbf{u}_k$. Mathematically, this is expressed as:

$$
J_k^\star(\mathbf{x}_k) = \min_{\mathbf{u}_k} \left[ c_k(\mathbf{x}_k, \mathbf{u}_k) + J_{k+1}^\star(\mathbf{f}_k(\mathbf{x}_k, \mathbf{u}_k)) \right]
$$

This equation is known as Bellman's equation, named after Richard Bellman, who formulated the principle of optimality:

> An optimal policy has the property that whatever the previous state and decision, the remaining decisions must constitute an optimal policy with regard to the state resulting from the previous decision.

In other words, any sub-path of an optimal path, from any intermediate point to the end, must itself be optimal.

Dynamic programming can handle nonlinear systems and non-quadratic cost functions naturally. It provides a global optimal solution, when one exists, and can incorporate state and control constraints with relative ease. Howver, as the dimension of the state space increases, this approach suffers from what Bellman termed the "curse of dimensionality." The computational complexity and memory requirements grow exponentially with the state dimension, rendering direct application of dynamic programming intractable for high-dimensional problems.

Fortunately, learning-based methods offer efficient tools to combat the curse of dimensionality on two fronts: by using function approximation (e.g., neural networks) to avoid explicit discretization, and by leveraging randomization through Monte Carlo methods inherent in the learning paradigm. Most of this course is dedicated to those ideas.

## Backward Recursion 

The principle of optimality provides a methodology for solving optimal control problems. Beginning at the final time horizon and working backwards, at each stage the local optimization problem given by Bellman's equation is solved. This process, termed backward recursion or backward induction constructs the optimal value function stage by stage.

````{prf:algorithm} Backward Recursion for Dynamic Programming
:label: backward-recursion

**Input:** Terminal cost function $c_T(\cdot)$, stage cost functions $c_t(\cdot, \cdot)$, system dynamics $f_t(\cdot, \cdot)$, time horizon $T$

**Output:** Optimal value functions $J_t^\star(\cdot)$ and optimal control policies $\mu_t^\star(\cdot)$ for $t = 1, \ldots, T$

1. Initialize $J_T^\star(\mathbf{x}) = c_T(\mathbf{x})$ for all $\mathbf{x}$ in the state space
2. For $t = T-1, T-2, \ldots, 1$:
   1. For each state $\mathbf{x}$ in the state space:
      1. Compute $J_t^\star(\mathbf{x}) = \min_{\mathbf{u}} \left[ c_t(\mathbf{x}, \mathbf{u}) + J_{t+1}^\star(f_t(\mathbf{x}, \mathbf{u})) \right]$
      2. Compute $\mu_t^\star(\mathbf{x}) = \arg\min_{\mathbf{u}} \left[ c_t(\mathbf{x}, \mathbf{u}) + J_{t+1}^\star(f_t(\mathbf{x}, \mathbf{u})) \right]$
   2. End For
3. End For
4. Return $J_t^\star(\cdot)$, $\mu_t^\star(\cdot)$ for $t = 1, \ldots, T$
````


Upon completion of this backward pass, we now have access to the optimal control to take at any stage and in any state. Furthermore, we can simulate optimal trajectories from any initial state and applying the optimal policy at each stage to generate the optimal trajectory.

## Example: Optimal Harvest in Resource Management

Dynamic programming is often used in resource management and conservation biology to devise policies to be implemented by decision makers and stakeholders : for eg. in fishereries, or timber harvesting. Per {cite}`Conroy2013`, we consider a population of a particular species, whose abundance we denote by $x_t$, where $t$ represents discrete time steps. Our objective is to maximize the cumulative harvest over a finite time horizon, while also considering the long-term sustainability of the population. This optimization problem can be formulated as:

$$
\text{maximize} \quad \sum_{t=t_0}^{t_f} F(x_t \cdot h_t) + F_T(x_{t_f})
$$

Here, $F(\cdot)$ represents the immediate reward function associated with harvesting, $h_t$ is the harvest rate at time $t$, and $F_T(\cdot)$ denotes a terminal value function that could potentially assign value to the final population state. In this particular problem, we assign no terminal value to the final population state, setting $F_T(x_{t_f}) = 0$ and allowing us to focus solely on the cumulative harvest over the time horizon.

In our model population model, the abundance of a specicy $x$ ranges from 1 to 100 individuals. The decision variable is the harvest rate $h$, which can take values from the set $D = \{0, 0.1, 0.2, 0.3, 0.4, 0.5\}$. The population dynamics are governed by a modified logistic growth model:

$$
x_{t+1} = x_t + 0.3x_t(1 - x_t/125) - h_tx_t
$$

where the $0.3$ represents the growth rate and $125$ is the carrying capacity (the maximum population size given the available resources). The logistic growth model returns continuous values; however our DP formulation uses a discrete state space. Therefore, we also round the the outcomes to the nearest integer.


Applying the principle of optimality, we can express the optimal value function $J^\star(x_t,t)$ recursively:

$$
J^\star(x_t, t) = \max_{h_t \in D} (F(x, h, t) + J^*(x_{t+1}, t+1))
$$

with the boundary condition $J^*(x_{t_f}) = 0$.

It's worth noting that while this example uses a relatively simple model, the same principles can be applied to more complex scenarios involving stochasticity, multiple species interactions, or spatial heterogeneity. 

```{code-cell} ipython3
:tags: [hide-input]
:load: code/harvest_dp.py

```
## Discretization and Interpolation

In many real-world problems, such as our resource management example, the state space is inherently continuous. However, the dynamic programming algorithm we've discussed operates on discrete state spaces. To bridge this gap, we have two main approaches: discretization and interpolation. In the previous example, we used a discretization method by rounding off values to the nearest grid point.

In our idealized models, we imagined population sizes as whole numbers—1 fish, 2 fish, 3 fish—but nature rarely conforms to such simplicity. What do you do when your survey reports 42.7 fish, for example? Without much explanation, our reflex in the previous example was to simply round things off. After all, what's the harm in calling 42.7 fish just 43? This approach, known as discretization, is the simplest way to handle continuous states. It's like overlaying a grid on a smooth landscape and only allowing yourself to stand at the intersections. In our demo code, we implemented this step via the [numpy.searchsorted](https://numpy.org/doc/2.0/reference/generated/numpy.searchsorted.html) function.

Discretization is straightforward and allows you to apply dynamic programming algorithms directly. For many problems, it might even be sufficient. However, as you might imagine from the various time discretization schemes we've encountered in trajectory optimization, we can do better. Specifically, we want to address the following issues:

1. Our model might make abrupt changes in decisions, even when the population barely shifts.
2. We're losing precision, especially when dealing with smaller population sizes where every individual counts.
3. We might want to scale up and add more factors to our model—perhaps considering the population's age structure or environmental variables. However, the curse of dimensionality might leave us needing an impossibly large number of grid points to maintain accuracy.

Rather than confining ourselves to grid intersections as we did with basic discretization, we can estimate the value between them via interpolation. When you encounter a state that doesn't exactly match a grid point—like that population of 42.7 fish—you can estimate its value based on nearby points you've already calculated. In its simplest form, we could use linear interpolation. Intuitively, it's like estimating the height of a hill between two surveyed points by drawing a straight line between them. Let's formalize this approach in the context of the backward induction procedure.

### Backward Recursion with Interpolation

In a continuous state space, we don't have $J_{k+1}^\star(\mathbf{x}_{k+1})$ directly available for all possible $\mathbf{x}_{k+1}$. Instead, we have $J_{k+1}^\star(\mathbf{x})$ for a set of discrete grid points $\mathbf{x} \in \mathcal{X}_\text{grid}$. We use interpolation to estimate $J_{k+1}^\star(\mathbf{x}_{k+1})$ for any $\mathbf{x}_{k+1}$ not in $\mathcal{X}_\text{grid}$.

Let's define an interpolation function $I_{k+1}(\mathbf{x})$ that estimates $J_{k+1}^\star(\mathbf{x})$ for any $\mathbf{x}$ based on the known values at grid points. Then, we can express Bellman's equation with interpolation as:

$$
J_k^\star(\mathbf{x}_k) = \min_{\mathbf{u}_k} \left[ c_k(\mathbf{x}_k, \mathbf{u}_k) + I_{k+1}(\mathbf{f}_k(\mathbf{x}_k, \mathbf{u}_k)) \right]
$$

For linear interpolation in a one-dimensional state space, $I_{k+1}(\mathbf{x})$ would be defined as:

$$
I_{k+1}(x) = J_{k+1}^\star(x_l) + \frac{x - x_l}{x_u - x_l} \left(J_{k+1}^\star(x_u) - J_{k+1}^\star(x_l)\right)
$$

where $x_l$ and $x_u$ are the nearest lower and upper grid points to $x$, respectively.

Here's a pseudo-code algorithm for backward recursion with interpolation:

````{prf:algorithm} Backward Recursion with Interpolation for Dynamic Programming
:label: backward-recursion-interpolation

**Input:** 
- Terminal cost function $c_T(\cdot)$
- Stage cost functions $c_t(\cdot, \cdot)$
- System dynamics $f_t(\cdot, \cdot)$
- Time horizon $T$
- Grid of state points $\mathcal{X}_\text{grid}$
- Set of possible actions $\mathcal{U}$

**Output:** Optimal value functions $J_t^\star(\cdot)$ and optimal control policies $\mu_t^\star(\cdot)$ for $t = 1, \ldots, T$ at grid points

1. Initialize $J_T^\star(\mathbf{x}) = c_T(\mathbf{x})$ for all $\mathbf{x} \in \mathcal{X}_\text{grid}$
2. For $t = T-1, T-2, \ldots, 1$:
   1. For each state $\mathbf{x} \in \mathcal{X}_\text{grid}$:
      1. Initialize $J_t^\star(\mathbf{x}) = \infty$ and $\mu_t^\star(\mathbf{x}) = \text{None}$
      2. For each action $\mathbf{u} \in \mathcal{U}$:
         1. Compute next state $\mathbf{x}_\text{next} = f_t(\mathbf{x}, \mathbf{u})$
         2. Compute interpolated future cost $J_\text{future} = I_{t+1}(\mathbf{x}_\text{next})$
         3. Compute total cost $J_\text{total} = c_t(\mathbf{x}, \mathbf{u}) + J_\text{future}$
         4. If $J_\text{total} < J_t^\star(\mathbf{x})$:
            1. Update $J_t^\star(\mathbf{x}) = J_\text{total}$
            2. Update $\mu_t^\star(\mathbf{x}) = \mathbf{u}$
   2. End For
3. End For
4. Return $J_t^\star(\cdot)$, $\mu_t^\star(\cdot)$ for $t = 1, \ldots, T$
````

### Implementation Considerations

The choice of interpolation method can significantly affect the accuracy of the solution. Linear interpolation is simple and often effective, but higher-order methods like cubic spline interpolation might provide better results in some problems. Furthermore, the layout and density of the grid points in $\mathcal{X}_\text{grid}$ can impact both the accuracy and computational efficiency. A finer grid generally provides better accuracy but increases computational cost. To balance this tradeoff, you might consider techniques like adaptive grid refinement or function approximation methods instead of fixed grid-based interpolation. Special care may also be needed for states near the boundaries, where interpolation might not be possible in all directions.

While simple to implement, interpolation methods scale poorly in multi-dimensional spaces in terms of computational complexity. Techniques like multilinear interpolation with tensorized representations or more advanced methods like radial basis function interpolation might be necessary.

To better address this computational challenge, we will broaden our perspective through the lens of numerical approximation methods for solving functional operator equations. Polynomial interpolation is a form of approximation, with properties akin to generalization in machine learning. By building these connections, we will develop techniques capable of more robustly handling the curse of dimensionality by leveraging the generalization properties of machine learning models, and the "blessing of randomness" inherent in supervised learning and Monte Carlo methods.

### Example: Optimal Harvest with Linear Interpolation

Here is a demonstration of the backward recursion procedure using linear interpolation. 

```{code-cell} ipython3
:tags: [hide-input]
:load: code/harvest_dp_linear_interpolation.py
```

Due to pedagogical considerations, this example is using our own implementation of the linear interpolation procedure. However, a more general and practical approach would be to use a built-in interpolation procedure in Numpy. Because our state space has a single dimension, we can simply use [scipy.interpolate.interp1d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.) which offers various interpolation methods through its `kind` argument, which can take values in 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'. 'zero', 'slinear', 'quadratic' and 'cubic'.

Here's a more general implementation which here uses cubic interpolation through the `scipy.interpolate.interp1d` function: 

```{code-cell} ipython3
:tags: [hide-input]
:load: code/harvest_dp_interp1d.py
```

## Linear Quadratic Regulator via Dynamic Programming

Let us now consider a special case of our dynamic programming formulation: the discrete-time Linear Quadratic Regulator (LQR) problem. This example will illustrate how the structure of linear dynamics and quadratic costs leads to a particularly elegant form of the backward recursion.

Consider a linear time-invariant system with dynamics:

$$
\mathbf{x}_{t+1} = A\mathbf{x}_t + B\mathbf{u}_t
$$

where $\mathbf{x}_t \in \mathbb{R}^n$ is the state and $\mathbf{u}_t \in \mathbb{R}^m$ is the control input.  
The cost function to be minimized is quadratic:

$$
J = \frac{1}{2}\mathbf{x}_T^\top S_T \mathbf{x}_T + \frac{1}{2}\sum_{t=0}^{T-1} \left(\mathbf{x}_t^\top Q \mathbf{x}_t + \mathbf{u}_t^\top R \mathbf{u}_t\right)
$$

where $S_T \geq 0$, $Q \geq 0$, and $R > 0$ are symmetric matrices of appropriate dimensions.  
Our goal is to find the optimal control sequence $\mathbf{u}_t^*$ that minimizes $J$ over a fixed time horizon $[0, T]$, given an initial state $\mathbf{x}_0$.

Let's apply the principle of optimality to derive the backward recursion for this problem. We'll start at the final time step and work our way backward.

At $t = T$, the terminal cost is given by:

$$
J_T^*(\mathbf{x}_T) = \frac{1}{2}\mathbf{x}_T^\top S_T \mathbf{x}_T
$$

At $t = T-1$, the cost-to-go is:

$$
J_{T-1}(\mathbf{x}_{T-1}, \mathbf{u}_{T-1}) = \frac{1}{2}\mathbf{x}_{T-1}^\top Q \mathbf{x}_{T-1} + \frac{1}{2}\mathbf{u}_{T-1}^\top R \mathbf{u}_{T-1} + J_T^*(\mathbf{x}_T)
$$

Substituting the dynamics equation:

$$
J_{T-1} = \frac{1}{2}\mathbf{x}_{T-1}^\top Q \mathbf{x}_{T-1} + \frac{1}{2}\mathbf{u}_{T-1}^\top R \mathbf{u}_{T-1} + \frac{1}{2}(A\mathbf{x}_{T-1} + B\mathbf{u}_{T-1})^\top S_T (A\mathbf{x}_{T-1} + B\mathbf{u}_{T-1})
$$

To find the optimal control, we differentiate with respect to $\mathbf{u}_{T-1}$ and set it to zero:

$$
\frac{\partial J_{T-1}}{\partial \mathbf{u}_{T-1}} = R\mathbf{u}_{T-1} + B^\top S_T (A\mathbf{x}_{T-1} + B\mathbf{u}_{T-1}) = 0
$$

Solving for $\mathbf{u}_{T-1}^*$:

$$
\mathbf{u}_{T-1}^* = -(R + B^\top S_T B)^{-1}B^\top S_T A\mathbf{x}_{T-1}
$$

We can define the gain matrix:

$$
K_{T-1} = (R + B^\top S_T B)^{-1}B^\top S_T A
$$

So that $\mathbf{u}_{T-1}^* = -K_{T-1}\mathbf{x}_{T-1}$. The optimal cost-to-go at $T-1$ is then:

$$
J_{T-1}^*(\mathbf{x}_{T-1}) = \frac{1}{2}\mathbf{x}_{T-1}^\top S_{T-1} \mathbf{x}_{T-1}
$$

Where $S_{T-1}$ is given by:

$$
S_{T-1} = Q + A^\top S_T A - A^\top S_T B(R + B^\top S_T B)^{-1}B^\top S_T A
$$

Continuing this process backward in time, we find that for any $t$:

$$
\mathbf{u}_t^* = -K_t\mathbf{x}_t
$$

Where:

$$
K_t = (R + B^\top S_{t+1} B)^{-1}B^\top S_{t+1} A
$$

And the optimal cost-to-go is:

$$
J_t^*(\mathbf{x}_t) = \frac{1}{2}\mathbf{x}_t^\top S_t \mathbf{x}_t
$$

Where $S_t$ satisfies the so-called discrete-time Riccati equation:

$$
S_t = Q + A^\top S_{t+1} A - A^\top S_{t+1} B(R + B^\top S_{t+1} B)^{-1}B^\top S_{t+1} A
$$
<!-- 
### Example: Linear Quadratic Regulation of a Liquid Tank 

We are dealing with a liquid-level control system for a storage tank. This system consists of a reservoir connected to a tank via valves. These valves are controlled by a gear train, which is driven by a DC motor. The motor, in turn, is controlled by an electronic amplifier. The goal is to maintain a constant liquid level in the tank, adjusting only when necessary.

The system is described by a third-order continuous-time model with the following state variables:
- $x_1(t)$: the height of the liquid in the tank
- $x_2(t)$: the angular position of the electric motor driving the valves
- $x_3(t)$: the angular velocity of the motor

The input to the system, $u(t)$, represents the signal sent to the electronic amplifier connected to the motor.
The system dynamics are described by the following differential equations:

$$
\begin{aligned}
& \dot{x}_1(t) = -2x_1(t) \\
& \dot{x}_2(t) = x_3(t) \\
& \dot{x}_3(t) = -10x_3(t) + 9000u(t)
\end{aligned}
$$

To pose this as a discrete-time LQR problem, we need to discretize the continuous-time system. Let's assume a sampling time of $T_s$ seconds. We can use the forward Euler method for a simple discretization:

$$
\begin{aligned}
& x_1(k+1) = x_1(k) + T_s(-2x_1(k)) \\
& x_2(k+1) = x_2(k) + T_sx_3(k) \\
& x_3(k+1) = x_3(k) + T_s(-10x_3(k) + 9000u(k))
\end{aligned}
$$

This can be written in the standard discrete-time state-space form:

$x(k+1) = Ax(k) + Bu(k)$

Where:

$x(k) = \begin{bmatrix} x_1(k) \\ x_2(k) \\ x_3(k) \end{bmatrix}$

$A = \begin{bmatrix} 
1-2T_s & 0 & 0 \\
0 & 1 & T_s \\
0 & 0 & 1-10T_s
\end{bmatrix}$

$B = \begin{bmatrix} 
0 \\
0 \\
9000T_s
\end{bmatrix}$

The goal of our LQR controller is to maintain the liquid level at a desired reference value while minimizing control effort. We can formulate this as a discrete-time LQR problem with the following cost function:

$J = \sum_{k=0}^{\infty} \left( (x_1(k) - x_{1,ref})^2 + ru^2(k) \right)$

Where $x_{1,ref}$ is the reference liquid level and $r$ is a positive weight on the control input.

To put this in standard discrete-time LQR form, we rewrite the cost function as:

$J = \sum_{k=0}^{\infty} \left( x^T(k)Qx(k) + ru^2(k) \right)$

Where:

$Q = \begin{bmatrix}
1 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}$

The discrete-time LQR problem is now to find the optimal control law $u^*(k) = -Kx(k)$ that minimizes this cost function, subject to the discrete-time system dynamics $x(k+1) = Ax(k) + Bu(k)$.

The solution involves solving the discrete-time algebraic Riccati equation:

$P = A^TPA - A^TPB(B^TPB + R)^{-1}B^TPA + Q$

Where $R = r$ (a scalar in this case), to find the positive definite matrix $P$. Then, the optimal gain matrix $K$ is given by:

$K = (B^TPB + R)^{-1}B^TPA$

This formulation ensures that:
1. The liquid level ($x_1(k)$) is maintained close to the reference value.
2. The system acts primarily when there's a change in the liquid level, as only $x_1(k)$ is directly penalized in the cost function.
3. The control effort is minimized, ensuring smooth operation of the valves.

By tuning the weight $r$ and the sampling time $T_s$, we can balance the trade-off between maintaining the desired liquid level, the amount of control effort used, and the responsiveness of the system. -->

## Stochastic Dynamic Programming

While our previous discussion centered on deterministic systems, many real-world problems involve uncertainty. Stochastic Dynamic Programming (SDP) extends our framework to handle stochasticity in both the objective function and system dynamics.

In the stochastic setting, our system evolution takes the form:

$$ \mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t) $$

Here, $\mathbf{w}_t$ represents a random disturbance or noise term at time $t$ due to the inherent uncertainty in the system's behavior. The stage cost function may also incorporate stochastic influences:

$$ c_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t) $$

In this context, our objective shifts from minimizing a deterministic cost to minimizing the expected total cost:

$$ \mathbb{E}\left[c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t)\right] $$

where the expectation is taken over the distributions of the random variables $\mathbf{w}_t$. The principle of optimality still holds in the stochastic case, but Bellman's optimality equation now involves an expectation:

$$ J_k^\star(\mathbf{x}_k) = \min_{\mathbf{u}_k} \mathbb{E}_{\mathbf{w}_k}\left[c_k(\mathbf{x}_k, \mathbf{u}_k, \mathbf{w}_k) + J_{k+1}^\star(\mathbf{f}_k(\mathbf{x}_k, \mathbf{u}_k, \mathbf{w}_k))\right] $$

In practice, this expectation is often computed by discretizing the distribution of $\mathbf{w}_k$ when the set of possible disturbances is very large or even continuous. Let's say we approximate the distribution with $K$ discrete values $\mathbf{w}_k^i$, each occurring with probability $p_k^i$. Then our Bellman equation becomes:

$$ J_k^\star(\mathbf{x}_k) = \min_{\mathbf{u}_k} \sum_{i=1}^K p_k^i \left(c_k(\mathbf{x}_k, \mathbf{u}_k, \mathbf{w}_k^i) + J_{k+1}^\star(\mathbf{f}_k(\mathbf{x}_k, \mathbf{u}_k, \mathbf{w}_k^i))\right) $$

The backward recursion algorithm for SDP follows a similar structure to its deterministic counterpart, with the key difference being that we now have to compute expected values: 

````{prf:algorithm} Backward Recursion for Stochastic Dynamic Programming
:label: stochastic-backward-recursion

**Input:** Terminal cost function $c_T(\cdot)$, stage cost functions $c_t(\cdot, \cdot, \cdot)$, system dynamics $\mathbf{f}_t(\cdot, \cdot, \cdot)$, time horizon $T$, disturbance distributions

**Output:** Optimal value functions $J_t^\star(\cdot)$ and optimal control policies $\mu_t^\star(\cdot)$ for $t = 1, \ldots, T$

1. Initialize $J_T^\star(\mathbf{x}) = c_T(\mathbf{x})$ for all $\mathbf{x}$ in the state space
2. For $t = T-1, T-2, \ldots, 1$:
   1. For each state $\mathbf{x}$ in the state space:
      1. Compute $J_t^\star(\mathbf{x}) = \min_{\mathbf{u}} \mathbb{E}_{\mathbf{w}_t}\left[c_t(\mathbf{x}, \mathbf{u}, \mathbf{w}_t) + J_{t+1}^\star(\mathbf{f}_t(\mathbf{x}, \mathbf{u}, \mathbf{w}_t))\right]$
      2. Compute $\mu_t^\star(\mathbf{x}) = \arg\min_{\mathbf{u}} \mathbb{E}_{\mathbf{w}_t}\left[c_t(\mathbf{x}, \mathbf{u}, \mathbf{w}_t) + J_{t+1}^\star(\mathbf{f}_t(\mathbf{x}, \mathbf{u}, \mathbf{w}_t))\right]$
   2. End For
3. End For
4. Return $J_t^\star(\cdot)$, $\mu_t^\star(\cdot)$ for $t = 1, \ldots, T$
````

While SDP provides us with a framework to for handling uncertainty, it makes the curse of dimensionality even more difficult to handle in practice. Not only does the state space need to be discretized, but now the disturbance space must be discretized as well. This can lead to a combinatorial explosion in the number of scenarios to be evaluated at each stage.

However, just as we tackled the challenges of continuous state spaces with discretization and interpolation, we can devise efficient methods to handle the additional complexity of evaluating expectations. This problem essentially becomes one of numerical integration. When the set of disturbances is continuous (as is often the case with continuous state spaces), we enter a domain where numerical quadrature methods could be applied. But these methods tend to scale poorly as the number of dimensions grows. This is where more efficient techniques, often rooted in Monte Carlo methods, come into play. The combination of two key ingredients emerges to tackle the curse of dimensionality:

1. Function approximation (through discretization, interpolation, neural networks, etc.)
2. Monte Carlo integration (simulation)

These two elements essentially distill the key ingredients of machine learning, which is the direction we'll be exploring in this course. 

### Example: Stochastic Optimal Harvest in Resource Management

Building upon our previous deterministic model, we now introduce stochasticity to more accurately reflect the uncertainties inherent in real-world resource management scenarios {cite:p}`Conroy2013`. As before, we consider a population of a particular species, whose abundance we denote by $x_t$, where $t$ represents discrete time steps. Our objective remains to maximize the cumulative harvest over a finite time horizon, while also considering the long-term sustainability of the population. However, we now account for two sources of stochasticity: partial controllability of harvest and environmental variability affecting growth rates.
The optimization problem can be formulated as:

$$
\text{maximize} \quad \mathbb{E}\left[\sum_{t=t_0}^{t_f} F(x_t \cdot h_t)\right]
$$

Here, $F(\cdot)$ represents the immediate reward function associated with harvesting, and $h_t$ is the realized harvest rate at time $t$. The expectation $\mathbb{E}[\cdot]$ over both harvest and growth rates, which we view as random variables. 
In our stochastic model, the abundance $x$ still ranges from 1 to 100 individuals. The decision variable is now the desired harvest rate $d_t$, which can take values from the set $D = {0, 0.1, 0.2, 0.3, 0.4, 0.5}$. However, the realized harvest rate $h_t$ is stochastic and follows a discrete distribution:

$$
h_t = \begin{cases}
0.75d_t & \text{with probability } 0.25 \\
d_t & \text{with probability } 0.5 \\
1.25d_t & \text{with probability } 0.25
\end{cases}
$$

By expressing the harvest rate as a random variable, we mean to capture the fact that harvesting is a not completely under our control: we might obtain more or less what we had intended to. Furthermore, we generalize the population dynamics to the stochastic cse via: 
$$

x_{t+1} = x_t + r_tx_t(1 - x_t/K) - h_tx_t
$$

where $K = 125$ is the carrying capacity. The growth rate $r_t$ is now stochastic and follows a discrete distribution:

$$
r_t = \begin{cases}
0.85r_{\text{max}} & \text{with probability } 0.25 \\
1.05r_{\text{max}} & \text{with probability } 0.5 \\
1.15r_{\text{max}} & \text{with probability } 0.25
\end{cases}
$$

where $r_{\text{max}} = 0.3$ is the maximum growth rate. 
Applying the principle of optimality, we can express the optimal value function $J^\star(x_t, t)$ recursively:

$$
J^\star(x_t, t) = \max_{d(t) \in D} \mathbb{E}\left[F(x_t \cdot h_t) + J^\star(x_{t+1}, t+1)\right]
$$

where the expectation is taken over the harvest and growth rate random variables. The boundary condition remains $J^*(x_{t_f}) = 0$. We can now adapt our previous code to account for the stochasticity in our model. One important difference is that simulating a solution in this context requires multiple realizations of our process. This is an important consideration when evaluating reinforcement learning methods in practice, as success cannot be claimed based on a single successful trajectory.

```{code-cell} ipython3
:tags: [hide-input]
:load: code/harvest_sdp.py
```

### Markov Decision Process Formulation

Rather than expressing the stochasticity in our system through a disturbance term as a parameter to a deterministic difference equation, we often work with an alternative representation (more common in operations research) which uses the Markov Decision Process formulation. The idea is that when we model our system in this way with the disturbance term being drawn indepently of the previous stages, the induced trajectory are those of a Markov chain. Hence, we can re-cast our control problem in that language, leading to the so-called Markov Decision Process framework in which we express the system dynamics in terms of transition probabilities rather than explicit state equations. In this framework, we express the probability that the system is in a given state using the transition probability function:

$$ p_t(\mathbf{x}_{t+1} | \mathbf{x}_t, \mathbf{u}_t) $$

This function gives the probability of transitioning to state $\mathbf{x}_{t+1}$ at time $t+1$, given that the system is in state $\mathbf{x}_t$ and action $\mathbf{u}_t$ is taken at time $t$. Therefore, $p_t$ specifies a conditional probability distribution over the next states: namely, the sum (for discrete state spaces) or integral over the next state should be 1.

Given the control theory formulation of our problem via a deterministic dynamics function and a noise term, we can derive the corresponding transition probability function through the following relationship:

$$
\begin{aligned}
p_t(\mathbf{x}_{t+1} | \mathbf{x}_t, \mathbf{u}_t) &= \mathbb{P}(\mathbf{W}_t \in \left\{\mathbf{w} \in \mathbf{W}: \mathbf{x}_{t+1} = f_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w})\right\}) \\
&= \sum_{\left\{\mathbf{w} \in \mathbf{W}: \mathbf{x}_{t+1} = f_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w})\right\}} q_t(\mathbf{w})
\end{aligned}
$$

Here, $q_t(\mathbf{w})$ represents the probability density or mass function of the disturbance $\mathbf{W}_t$ (assuming discrete state spaces). When dealing with continuous spaces, the above expression simply contains an integral rather than a summation. 


For a system with deterministic dynamics and no disturbance, the transition probabilities become much simpler and be expressed using the indicator function. Given a deterministic system with dynamics:

$$ \mathbf{x}_{t+1} = f_t(\mathbf{x}_t, \mathbf{u}_t) $$

The transition probability function can be expressed as:

$$ p_t(\mathbf{x}_{t+1} | \mathbf{x}_t, \mathbf{u}_t) = \begin{cases}
1 & \text{if } \mathbf{x}_{t+1} = f_t(\mathbf{x}_t, \mathbf{u}_t) \\
0 & \text{otherwise}
\end{cases} $$

With this transition probability function, we can recast our Bellman optimality equation:

$$ J_t^\star(\mathbf{x}_t) = \max_{\mathbf{u}_t \in \mathbf{U}} \left\{ c_t(\mathbf{x}_t, \mathbf{u}_t) + \sum_{\mathbf{x}_{t+1}} p_t(\mathbf{x}_{t+1} | \mathbf{x}_t, \mathbf{u}_t) J_{t+1}^\star(\mathbf{x}_{t+1}) \right\} $$

Here, ${c}(\mathbf{x}_t, \mathbf{u}_t)$ represents the expected immediate reward (or negative cost) when in state $\mathbf{x}_t$ and taking action $\mathbf{u}_t$ at time $t$. The summation term computes the expected optimal value for the future states, weighted by their transition probabilities.

This formulation offers several advantages:

1. It makes the Markovian nature of the problem explicit: the future state depends only on the current state and action, not on the history of states and actions.

2. For discrete-state problems, the entire system dynamics can be specified by a set of transition matrices, one for each possible action.

3. It allows us to bridge the gap with the wealth of methods in the field of probabilistic graphical models and statistical machine learning techniques for modelling and analysis. 

## From Control Theory to Operations Research Notation

The presentation above was intended to bridge the gap between the control-theoretic perspective and the world of closed-loop control through the idea of determining the value function of a parametric optimal control problem. We then saw how the backward induction procedure was applicable to both the deterministic and stochastic cases by taking the expectation over the disturbance variable. We then said that we can alternatively work with a representation of our system where instead of writing our model as a deterministic dynamics function taking a disturbance as an input, we would rather work directly via its transition probability function, which gives rise to the Markov chain interpretation of our system in simulation.

Now we should highlight that the notation used in control theory tends to differ from that found in operations research communities, in which the field of dynamic programming flourished. We summarize those (purely notational) differences in this section.

In operations research, the system state at each decision epoch is typically denoted by $s \in \mathcal{S}$, where $S$ is the set of possible system states. When the system is in state $s$, the decision maker may choose an action $a$ from the set of allowable actions $\mathcal{A}_s$. The union of all action sets is denoted as $\mathcal{A} = \bigcup_{s \in \mathcal{S}} \mathcal{A}_s$.

The dynamics of the system are described by a transition probability function $p_t(j | s, a)$, which represents the probability of transitioning to state $j \in \mathcal{S}$ at time $t+1$, given that the system is in state $s$ at time $t$ and action $a \in \mathcal{A}_s$ is chosen. This transition probability function satisfies:

$$\sum_{j \in \mathcal{S}} p_t(j | s, a) = 1$$

It's worth noting that in operations research, we typically work with reward maximization rather than cost minimization, which is more common in control theory. However, we can easily switch between these perspectives by simply negating the quantity. That is, maximizing a reward function is equivalent to minimizing its negative, which we would then call a cost function.

The reward function is denoted by $r_t(s, a)$, representing the reward received at time $t$ when the system is in state $s$ and action $a$ is taken. In some cases, the reward may also depend on the next state, in which case it is denoted as $r_t(s, a, j)$. The expected reward can then be computed as:

$$r_t(s, a) = \sum_{j \in \mathcal{S}} r_t(s, a, j) p_t(j | s, a)$$

Combined together, these elemetns specify a Markov decision process, which is fully described by the tuple:

$$\{T, S, \mathcal{A}_s, p_t(\cdot | s, a), r_t(s, a)\}$$

where $T$ represents the set of decision epochs (the horizon).


### Example: Sample Size Determination in Pharmaceutical Development

Pharmaceutical development is the process of bringing a new drug from initial discovery to market availability. This process is lengthy, expensive, and risky, typically involving several stages:

1. **Drug Discovery**: Identifying a compound that could potentially treat a disease.
2. **Preclinical Testing**: Laboratory and animal testing to assess safety and efficacy.
. **Clinical Trials**: Testing the drug in humans, divided into phases:
   - Phase I: Testing for safety in a small group of healthy volunteers.
   - Phase II: Testing for efficacy and side effects in a larger group with the target condition.
   - Phase III: Large-scale testing to confirm efficacy and monitor side effects.
4. **Regulatory Review**: Submitting a New Drug Application (NDA) for approval.
5. **Post-Market Safety Monitoring**: Continuing to monitor the drug's effects after market release.

This process can take 10-15 years and cost over $1 billion {cite}`Adams2009`. The high costs and risks involved call for a principled approach to decision making. We'll focus on the clinical trial phases and NDA approval, per the MDP model presented by {cite}`Chang2010`:

1. **States** ($S$): Our state space is $S = \{s_1, s_2, s_3, s_4\}$, where:
   - $s_1$: Phase I clinical trial
   - $s_2$: Phase II clinical trial
   - $s_3$: Phase III clinical trial
   - $s_4$: NDA approval

2. **Actions** ($A$): At each state, the action is choosing the sample size $n_i$ for the corresponding clinical trial. The action space is $A = \{10, 11, ..., 1000\}$, representing possible sample sizes.

3. **Transition Probabilities** ($P$): The probability of moving from one state to the next depends on the chosen sample size and the inherent properties of the drug.
      We define:

   - $P(s_2|s_1, n_1) = p_{12}(n_1) = \sum_{i=0}^{\lfloor\eta_1 n_1\rfloor} \binom{n_1}{i} p_0^i (1-p_0)^{n_1-i}$
     where $p_0$ is the true toxicity rate and $\eta_1$ is the toxicity threshold for Phase I.
     
  - Of particular interest is the transition from Phase II to Phase III which we model as:

    $P(s_3|s_2, n_2) = p_{23}(n_2) = \Phi\left(\frac{\sqrt{n_2}}{2}\delta - z_{1-\eta_2}\right)$

    where $\Phi$ is the cumulative distribution function (CDF) of the standard normal distribution:

      $\Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x e^{-t^2/2} dt$

    This is giving us the probability that we would observe a treatment effect this large or larger if the null hypothesis (no treatment effect) were true. A higher probability indicates stronger evidence of a treatment effect, making it more likely that the drug will progress to Phase III.

    In this expression, $\delta$ is called the "normalized treatment effect". In clinical trials, we're often interested in the difference between the treatment and control groups. The "normalized" part means we've adjusted this difference for the variability in the data. Specifically $\delta = \frac{\mu_t - \mu_c}{\sigma}$ where $\mu_t$ is the mean outcome in the treatment group, $\mu_c$ is the mean outcome in the control group, and $\sigma$ is the standard deviation of the outcome. A larger $\delta$ indicates a stronger treatment effect.

    Furthermore, the term $z_{1-\eta_2}$ is the $(1-\eta_2)$-quantile of the standard normal distribution. In other words, it's the value where the probability of a standard normal random variable being greater than this value is $\eta_2$. For example, if $\eta_2 = 0.05$, then $z_{1-\eta_2} \approx 1.645$. A smaller $\eta_2$ makes the trial more conservative, requiring stronger evidence to proceed to Phase III.

    Finally, $n_2$ is the sample size for Phase II. The $\sqrt{n_2}$ term reflects that the precision of our estimate of the treatment effect improves with the square root of the sample size.

   - $P(s_4|s_3, n_3) = p_{34}(n_3) = \Phi\left(\frac{\sqrt{n_3}}{2}\delta - z_{1-\eta_3}\right)$
     where $\eta_3$ is the significance level for Phase III.

4. **Rewards** ($R$): The reward function captures the costs of running trials and the potential profit from a successful drug:

   - $R(s_i, n_i) = -c_i(n_i)$ for $i = 1, 2, 3$, where $c_i(n_i)$ is the cost of running a trial with sample size $n_i$.
   - $R(s_4) = g_4$, where $g_4$ is the expected profit from a successful drug.

5. **Discount Factor** ($\gamma$): We use a discount factor $0 < \gamma \leq 1$ to account for the time value of money and risk preferences.

```{code-cell} ipython3
:tags: [hide-input]
:load: code/sample_size_drug_dev_dp.py
```

## Decision Rules and Policies


In the presentation provided so far, we directly assumed that the form of our feedback controller was of the form $u(\mathbf{x}, t)$. The idea is that rather than just looking at the stage as in open-loop control, we would now consider the current state to account for the presence of noise. We came to that conclusion by considering the parametric optimization problem corresponding to the trajectory optimization perspective and saw that the "argmax" counterpart to the value function (the max) was exactly this function $u(x, t)$. But this presentation was mostly for intuition and neglected the fact that we could consider other kinds of feedback controllers. In the context of MDPs and under the OR terminology, we should now rather talk of policies instead of controllers.

But to properly introduce the concept of policy, we first have to talk about decision rules. A decision rule is a prescription of a procedure for action selection in each state at a specified decision epoch. These rules can vary in their complexity due to their potential dependence on the history and ways in which actions are then selected. Decision rules can be classified based on two main criteria:

1. Dependence on history: Markovian or History-dependent
2. Action selection method: Deterministic or Randomized

Markovian decision rules are those that depend only on the current state, while history-dependent rules consider the entire sequence of past states and actions. Formally, we can define a history $h_t$ at time $t$ as:

$$h_t = (s_1, a_1, \ldots, s_{t-1}, a_{t-1}, s_t)$$

where $s_u$ and $a_u$ denote the state and action at decision epoch $u$. The set of all possible histories at time $t$, denoted $H_t$, grows rapidly with $t$:

$$H_1 = \mathcal{S}$$
$$H_2 = \mathcal{S} \times A \times \mathcal{S}$$
$$H_t = H_{t-1} \times A \times \mathcal{S} = \mathcal{S} \times (A \times \mathcal{S})^{t-1}$$

This exponential growth in the size of the history set motivates us to seek conditions under which we can avoid searching for history-dependent decision rules and instead focus on Markovian rules, which are much simpler to implement and evaluate.

Decision rules can be further classified as deterministic or randomized. A deterministic rule selects an action with certainty, while a randomized rule specifies a probability distribution over the action space.

These classifications lead to four types of decision rules:
1. Markovian Deterministic (MD): $d_t: \mathcal{S} \rightarrow \mathcal{A}_s$
2. Markovian Randomized (MR): $d_t: \mathcal{S} \rightarrow \mathcal{P}(\mathcal{A}_s)$
3. History-dependent Deterministic (HD): $d_t: H_t \rightarrow \mathcal{A}_s$
4. History-dependent Randomized (HR): $d_t: H_t \rightarrow \mathcal{P}(\mathcal{A}_s)$

Where $\mathcal{P}(\mathcal{A}_s)$ denotes the set of probability distributions over $\mathcal{A}_s$.

<!-- The choice of decision rule affects how the MDP's rewards and transition probabilities are computed. For example, with a Markovian deterministic rule $d_t \in D_t^{MD}$, we have:

$$r_t(s, d_t(s)) \text{ and } p_t(j|s, d_t(s))$$

With a randomized Markovian rule, these become expected values:

$$r_t(s, d_t(s)) = \sum_{a \in \mathcal{A}_s} r_t(s,a) q_{d_t(s)}(a)$$
$$p_t(j|s, d_t(s)) = \sum_{a \in \mathcal{A}_s} p_t(j|s,a) q_{d_t(s)}(a)$$

Where $q_{d_t(s)}(a)$ is the probability of choosing action $a$ in state $s$ under decision rule $d_t$. -->

It's important to note that decision rules are stage-wise objects. However, to solve an MDP, we need a strategy for the entire horizon. This is where we make a distinction and introduce the concept of a policy. A policy $\pi$ is a sequence of decision rules, one for each decision epoch:

$$\pi = (d_1, d_2, ..., d_{N-1})$$

Where $N$ is the horizon length (possibly infinite). The set of all policies of class $K$ (where $K$ can be HR, HD, MR, or MD) is denoted as $\Pi^K$.

A special type of policy is a stationary policy, where the same decision rule is used at all epochs: $\pi = (d, d, ...)$, often denoted as $d^\infty$. Stationary policies are particularly important in infinite horizon problems.

The relationships between these policy classes form a hierarchy:

$$\begin{align*}
\Pi^{SD} \subset \Pi^{SR} \subset \Pi^{MR} \subset \Pi^{HR}\\
\Pi^{SD} \subset \Pi^{MD} \subset \Pi^{MR} \subset \Pi^{HR} \\
\Pi^{SD} \subset \Pi^{MD} \subset \Pi^{HD} \subset \Pi^{HR}
\end{align*}
$$

Where SD stands for Stationary Deterministic and SR for Stationary Randomized. The largest set is by far the set of history randomized policies. 

A fundamental question in MDP theory is: under what conditions can we avoid working with the set $\Pi^{HR}$ and focus for example on the much simpler set of deterministic Markovian policy? Even more so, we will see that in the infinite horizon case, we can drop the dependance on time and simply consider stationary deterministic Markovian policies. 

<!-- ### Special Kind of MDP: Bandit Models

Markov Decision Processes (MDPs) and optimal control problems in general are all about dealing with sequential problems: problems in which entire sequences of actions or ways of acting through time are considered. This makes these methods general and powerful, but also challenging from a practical and theoretical perspective.

In some problems, there might be further simplifications at play that would allow us to devise more efficient methods. A particular sub-class of such problems is that of bandit models. These models are often encountered in applications like adaptive optics, ad placement, or can even be found in frameworks like that of GFlowNets or LLMs.

More precisely, we define a bandit model as follows:

A bandit model is a sequential decision model in which, at each decision epoch, the decision maker observes the state of each of $K$ Markov reward processes and, based on the states, the transition probabilities, and rewards of each, selects a process to use in the current period. The selected process changes state according to its transition probabilities, and the states of all other processes remain fixed.

Formally, we can define a bandit model as a Markov Decision Process with the following components:

1. Decision epochs: $T = \{1, 2, \ldots, N\}$, where $N \leq \infty$.

2. States: $S = S^1 \times S^2 \times \cdots \times S^K$, where $S^i$ is the state space of the $i$-th process.

3. Actions: At each decision epoch, the action is to choose one of the $K$ processes.

4. Transition probabilities: For process $i$ in state $s^i \in S^i$, the transition probability to state $j^i$ is given by $p_t^i(j^i | s^i)$.

5. Rewards: When process $i$ in state $s^i$ is chosen, the decision maker receives a reward $r_t^i(s^i)$.

The effects of choosing process $i$ when it is in state $s^i \in S^i$, $i = 1, 2, \ldots, K$ are:

1. It changes state according to the transition law $p_t^i(j^i | s^i)$.
2. The decision maker receives a reward $r_t^i(s^i)$.
3. All other processes remain in their current state. -->