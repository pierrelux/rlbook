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


## Example: Sample Size Determination in Pharmaceutical Development

Pharmaceutical development is the process of bringing a new drug from initial discovery to market availability. This process is lengthy, expensive, and risky, typically involving several stages:

1. **Drug Discovery**: Identifying a compound that could potentially treat a disease.
2. **Preclinical Testing**: Laboratory and animal testing to assess safety and efficacy.
. **Clinical Trials**: Testing the drug in humans, divided into phases:
   - Phase I: Testing for safety in a small group of healthy volunteers.
   - Phase II: Testing for efficacy and side effects in a larger group with the target condition.
   - Phase III: Large-scale testing to confirm efficacy and monitor side effects.
4. **Regulatory Review**: Submitting a New Drug Application (NDA) for approval.
5. **Post-Market Safety Monitoring**: Continuing to monitor the drug's effects after market release.

This process can take 10-15 years and cost over $1 billion {cite:t}`Adams2009`. The high costs and risks involved call for a principled approach to decision making. We'll focus on the clinical trial phases and NDA approval, per the MDP model presented by {cite}`Chang2010`:

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
