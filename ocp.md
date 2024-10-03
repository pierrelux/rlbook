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

# Discrete-Time Trajectory Optimization


```{epigraph}
The sciences do not try to explain, they hardly even try to interpret, they mainly make models. By a model is meant a mathematical construct which, with the addition of certain verbal interpretations, describes observed phenomena. The justification of such a mathematical construct is solely and precisely that it is expected to work.

-- John von Neumann
```

This course considers two broad categories of environments, each with its own specialized solution methods: deterministic and stochastic environments. Stochastic problems are mathematically more general than their deterministic counterparts. However, despite this generality, it's important to note that algorithms for stochastic problems are not necessarily more powerful than those designed for deterministic ones when used in practice. We should keep in mind that stochasticity and determinism are assumed properties of the world, which we model—perhaps imperfectly—into our algorithms. In this course, we adopt a pragmatic perspective on that matter, and will make assumptions only to the extent that those assumptions help us design algorithms which are ultimately useful in practice: a lesson which we have certainly learned from the success of deep learning methods over the last years. With this pragmatic stance, we start our journey with deterministic discrete-time models. 

## Irrigation Management Example

Resource allocation problems, found across various fields of operations research, are a specific kind of deterministic discrete-time optimal control problems. For example, consider the problem of irrigation management as posed by {cite:t}`Hall1968`, in which a decision maker is tasked with finding the optimal amount of water to allocate throughout the various growth stages of a plant in order to maximize the yield. Clearly, allocating all the water -- thereby flooding the crop -- in the first stage would be a bad idea. Similarly, letting the crop dry out and only watering at the end is also suboptimal. Our solution -- that is a prescription for the amount of water to use at any point in time -- should balance those two extremes. In order to achieve this goal, our system should ideally be informed by the expected growth dynamics of the given crops as a function of the water provided: a process which depends at the very least on physical properties known to influence the soil moisture. The basic model considered in this paper describes the evolution of the moisture content through the equation: 

\begin{align*}
w_{t+1} = w_t+\texttip{\eta}{efficiency} u_t-e_t + \phi_t
\end{align*}

where $\eta$ is a fixed "efficiency" constant determining the soil moisture response to irrigation, and $e_t$ is a known quantity summarizing the effects of water loss due to evaporation and finally $\phi_t$ represents the expected added moisture due to precipitation. 

Furthermore, we should ensure that the total amount of water used throughout the season does not exceed the maximum allowed amount. To avoid situations where our crop receives too little or too much water, we further impose the condition that $w_p \leq w_{t+1} \leq w_f$ for all stages, for some given values of the so-called permanent wilting percentage $w_p$ and field capacity $w_f$. Depending on the moisture content of the soil, the yield will vary up to a maximum value $Y_max$ depending on the water deficiencies incurred throughout the season. The authors make the assumptions that such deficiencies interact multiplicatively across stages such that the total yield is given by
$\left[\prod_{t=1}^N d_t(w_t)\right] Y_{\max}$. Due to the operating cost of watering operations (for example, energy consumption of the pumps, human labor, etc), a more meaningful objective is to maximize $\prod_{t=1}^N d_t\left(w_t\right) Y_{\max } - \sum_{t=1}^N c_t\left(u_t\right)$. 
 The problem specification laid out above can be turned into the following mathematical program:

\begin{alignat*}{2}
\text{minimize} \quad & \sum_{t=1}^N c_t(u_t) - a_N Y_{\max} & \\
\text{such that} \quad 
& w_{t+1} = w_t + \eta u_t - e_t + \phi_t, & \quad & t = 1, \dots, N-1, \\
& q_{t+1} = q_t - u_t, & \quad & t = 1, \dots, N-1, \\
& a_{t+1} = d_t(w_t) a_t, & \quad & t = 1, \dots, N-1, \\
& w_p \leq w_{t} \leq w_f, & \quad & t = 1, \dots, N, \\
& 0 \leq u_t \leq q_t, & \quad & t = 1, \dots, N, \\
& 0 \leq q_t, & \quad & t = 1, \dots, N, \\
& a_0 = 1, & & \\
\text{given} \quad & w_1, q_N. &
\end{alignat*}

The multiplicative form of the objective function coming from the yield term has been eliminated through by adding a new variable, $a_t$, representing the product accumulation of water deficiencies since the beginning of the season. 

Clearly, this model is a simplification of real phenomena at play: that of the physical process of water absorption by a plant through its root system. Many more aspects of the world would have to be included to have a more faithful reproduction of the real process: for example by taking into account the real-time meteorological data, pressure, soil type, solar irradiance, shading and topology of the terrain etc. We could go to the level of even modelling the inner workings of the plant itself to understand exactly how much water will get absorbed. More crucially, our assumption that the water absorption takes place instantaneously at discrete points in time is certainly not true. So should we go back to the drawing board and consider a better model? The answer is that "it depends". It depends on the adequacy of the solution when deployed in the real world, or whether it helped provide insights to the user. Put simply: is the system useful to those who interact with it? Answering this question requires us to think more broadly about our system and how it will interact more broadly with the end users and the society.


# Mathematical Programming Formulation

The three quantities $(w_t, q_t, a_t)$ appearing in the model of {cite:t}`Hall1968` above have the property that they encompass all of the information necessary to simulate the process. We say that it is a "**state variable**", and its time evolution is specified via so-called **dynamics function**, which we commonly denote by $f_t$. In discrete-time systems, the dynamics are often described by "difference equations," as opposed to the "differential equations" used for continuous-time systems. When the dynamics function depends on the time index $t$, we refer to it as "non-stationary" dynamics. Conversely, if the function $f$ remains constant across all time steps, we call it "stationary" dynamics. In the context of optimal control theory, we refer more generally to $u_t$ as a "control" variable while in other communities it is called an "action". Whether the problem is posed as a minimization problem or maximization problem is also a matter of communities, with control theory typically posing problems in terms of cost minimization while OR and RL communities usually adopt a reward maximization perspective. In this course, we will alternate between the two equivalent formulations while ensuring that context is sufficiently clear to understand which one is used. 

The problem stated above, is known as a Discrete-Time Optimal Control Problem (DOCP), which we write more generically as: 

````{prf:definition} Discrete-Time Optimal Control Problem of Bolza Type
:label: bolza-docp
\begin{alignat*}{2}
\text{minimize} \quad & c_T(\mathbf{x}_T) + \sum_{t=1}^T c_t(\mathbf{x}_t, \mathbf{u}_t) & \\
\text{such that} \quad 
& \mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t), & \quad & t = 1, \dots, T-1, \\
& \mathbf{u}_{lb} \leq \mathbf{u}_t \leq \mathbf{u}_{ub}, & \quad & t = 1, \dots, T, \\
& \mathbf{x}_{lb} \leq \mathbf{x}_t \leq \mathbf{x}_{ub}, & \quad & t = 1, \dots, T, \\
\text{given} \quad & \mathbf{x}_1. &
\end{alignat*}
````
The objective function (sometimes called a "performance index") in a Bolza problem comprises of two terms: the sum of immediate cost or rewards per stage, and a terminal cost function (sometimes called "scrap value"). If the terminal cost function is omitted from the objective function, then the resulting DOCP is said to be of Lagrange type. 

````{prf:definition} Discrete-Time Optimal Control Problem of Lagrange Type
:label: lagrange-docp
\begin{alignat*}{2}
\text{minimize} \quad & \sum_{t=1}^T c_t(\mathbf{x}_t, \mathbf{u}_t) & \\
\text{such that} \quad 
& \mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t), & \quad & t = 1, \dots, T-1, \\
& \mathbf{u}_{lb} \leq \mathbf{u}_t \leq \mathbf{u}_{ub}, & \quad & t = 1, \dots, T, \\
& \mathbf{x}_{lb} \leq \mathbf{x}_t \leq \mathbf{x}_{ub}, & \quad & t = 1, \dots, T, \\
\text{given} \quad & \mathbf{x}_1. &
\end{alignat*}
````
Finally, a Mayer problem is one in which the objective function only contains a terminal cost function without explicitly accounting for immediate costs encountered across stages: 
````{prf:definition} Discrete-Time Optimal Control Problem of Mayer Type
:label: mayer-docp
\begin{alignat*}{2}
\text{minimize} \quad & c_T(\mathbf{x}_T) & \\
\text{such that} \quad 
& \mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t), & \quad & t = 1, \dots, T-1, \\
& \mathbf{u}_{lb} \leq \mathbf{u}_t \leq \mathbf{u}_{ub}, & \quad & t = 1, \dots, T, \\
& \mathbf{x}_{lb} \leq \mathbf{x}_t \leq \mathbf{x}_{ub}, & \quad & t = 1, \dots, T, \\
\text{given} \quad & \mathbf{x}_1. &
\end{alignat*}
````

When writing the optimal control problem in any of those three forms, it is implied that both $u_1, ..., u_T$ and the state trajectory $x_1, ..., x_T$ are optimization variables. Since we ultimately care about the decisions themselves, the idea of posing the states themselves as optimization variables seems uncessary given that we have access to the dynamics. We will soon see that there indeed exists a way in which we get rid of the state variables as constraints through a process of explicit simulation with the class of "shooting methods", thereby turning what would otherwise be an constrained optimization problem into an unconstrained one.  

## Reduction to Mayer Problems

While it might appear at first glance that Bolza problems are somehow more expressive or powerful, we can show that both Lagrange and Bolza problems can be reduced to a Mayer problem through the idea of "state augmentation". 
The overall idea is that the explicit sum of costs can be eliminated by maintaining a running sum of costs as an additional state variable $y_t$. The augmented system equation then becomes: 

\begin{align*}
\boldsymbol{\tilde{f}}_t\left(\boldsymbol{\boldsymbol{\tilde{x}}}_t, \boldsymbol{u}_t\right) \triangleq \left(\begin{array}{c}
\boldsymbol{f}_t\left(\boldsymbol{x}_t, \boldsymbol{u}_t\right) \\
y_t+c_t\left(\boldsymbol{x}_t, \boldsymbol{u}_t\right)
\end{array}\right) 
\end{align*}

where $\boldsymbol{\tilde{x}}_t \triangleq (\mathbf{x}_t, y_t)$ is the concatenation of the running cost so far and the underlying state of the original system. The terminal cost function in the Bolza-to-Mayer transformation is computed with:
\begin{align*}
\tilde{c}_T(\mathbf{\tilde{x}}_T)  \triangleq c_T\left(\boldsymbol{x}_T\right)+y_T
\end{align*}

This transformation is often helpful to simplify mathematical derivations (as we are about to see shortly) but could also be used to streamline algorithmic implementation (by maintaining one version of the code rather than three with many if/else statements). That being said, there could also be computational advantages to working with the specific problem types rather than the one size-fits-for-all Mayer reduction.

# Numerical Methods for Solving DOCPs

Let's assume that an optimal control problem has been formulated in one of the forms presented earlier and has been given to us to solve. The following section explores numerical solutions applicable to these problems, focusing on trajectory optimization. Our goal is to output an optimal control (and state trajectory) based on the given cost function and dynamics structure. It's important to note that the methods presented here are not learning methods just yet; they don't involve ingesting data or inferring unknown quantities from it. However, these methods represent a central component of any decision-learning system, and we will later explore how learning concepts can be incorporated.

Before delving into the solution methods, let's consider an electric vehicle energy management problem which we will use this as a test bed throughout this section. Consider an electric vehicle traversing a planned route, where we aim to optimize its energy consumption over a 20-minute journey. Our simplified model represents the vehicle's state using two variables: $x_1$, the battery state of charge as a percentage, and $x_2$, denoting the vehicle's speed in meters per second. The control input $u$, ranging from -1 to 1, represents the motor power, with negative values indicating regenerative braking and positive values representing acceleration. The problem can be formally expressed as a mathematical program in Bolza form:

\begin{align*}
\min_{\mathbf{x}, \mathbf{u}} \quad & J = \underbrace{x_{T,1}^2 + x_{T,2}^2}_{\text{Mayer term}} + \underbrace{\sum_{t=1}^{T-1} 0.1(x_{t,1}^2 + x_{t,2}^2 + u_t^2)}_{\text{Lagrange term}} \\
\text{subject to:} \quad & x_{t+1} = f_t(x_t, u_t), \quad t = 1, \ldots, T-1 \\
& x_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \\
& -1 \leq u_t \leq 1, \quad t = 1, \ldots, T-1 \\
& -5 \leq x_{t,1}, x_{t,2} \leq 5, \quad t = 1, \ldots, T \\
\text{where:} \quad & f_t(x_t, u_t) = \begin{bmatrix}
   x_{t,1} + 0.1x_{t,2} + 0.05u_t \\
   x_{t,2} + 0.1u_t
   \end{bmatrix} \\
& T = 20 \\
& x_t = \begin{bmatrix} x_{t,1} \\ x_{t,2} \end{bmatrix} \in \mathbb{R}^2, \quad u_t \in \mathbb{R}.
\end{align*}

The system dynamics, represented by $f_t(x_t, u_t)$, describe how the battery charge and vehicle speed evolve based on the current state and control input. The initial condition $x_1 = [1, 0]^T$ indicates that the vehicle starts with a fully charged battery and zero initial speed. The constraints $-1 \leq u_t \leq 1$ and $-5 \leq x_{t,1}, x_{t,2} \leq 5$ ensure that the control inputs and state variables remain within acceptable ranges throughout the journey. This model is of course highly simplistic and neglects the nonlinear nature of battery discharge and vehicle motion due to air resistance, road grade, and vehicle mass, etc. Furthermore, our model ignores the effect of environmental factors like wind and temperature on regenerative breaking. Route-specific information such as elevation changes and speed limits are absent, as is the consideration of auxiliary power consumption such as heating and entertainment. These are all possible improvements to our models which we ignore at the moment for the sake of simplicity.

## Single Shooting Methods

Given access to unconstrained optimization solver, the easiest method to implement is by far what is known as "single shooting" in control theory. The idea of simple: rather than having to solve for the state variables as equality constraints, we transform the original constrained problem into an unconstrained one through "simulation", ie by recursively computing the evolution of our system for any given set of controls and initial state. In the deterministic setting, given an initial state, we can always exactly reconstruct the resulting sequence of states by "rolling out" our model, a process which some communities would refer to as "time marching". Mathematically, this amounts to forming the following unconstrained program: 

$$
\begin{align*}
\min_{\substack{\mathbf{u}_1, \ldots, \mathbf{u}_{T-1} \\ \mathbf{u}_{lb} \leq \mathbf{u}_t \leq \mathbf{u}_{lb} \, t=1,...,T-1}} \quad c_T(\boldsymbol{\phi}_{T}(\boldsymbol{u}_{1:T-1}, \boldsymbol{x}_1)) + \sum_{t=1}^{T-1} c_t(\boldsymbol{\phi}_t(\boldsymbol{u}_{1:T-1}, \boldsymbol{x}_1), \boldsymbol{u}_{t})
\end{align*}
$$

To implement this transform, we construct a set of helper functions $\boldsymbol{\phi}_1, ..., \boldsymbol{\phi}_{T-1}$ whose role is compute the state at any time $t=1, ..., T$ resulting from applying the sequence of controls starting from the initial state. We can define those functions recursively as 

$$
\begin{align*}
&\boldsymbol{\phi}_t(\boldsymbol{u}_{1:T-1}, \boldsymbol{x}_1) \triangleq \boldsymbol{f}_{t-1}(\boldsymbol{\phi}_{t-1}(\boldsymbol{u}_{1:T-1}, \boldsymbol{x}_1), \boldsymbol{u}_{t-1}), \quad t=2,...,T\\
&\text{with}\quad \boldsymbol{\phi}_1(\boldsymbol{u}_{1:T}, \boldsymbol{x}_1) \triangleq \boldsymbol{x}_1
\end{align*}
$$

```{prf:algorithm} Naive Single Shooting: re-computation/checkpointing
:label: naive-single-shooting

**Inputs** Initial state $\mathbf{x}_1$, time horizon $T$, control bounds $\mathbf{u}_{lb}$ and $\mathbf{u}_{ub}$, state transition functions $\mathbf{f}_t$, cost functions $c_t$

**Output** Optimal control sequence $\mathbf{u}^*_{1:T-1}$

1. Initialize $\mathbf{u}_{1:T-1}$ within bounds $[\mathbf{u}_{lb}, \mathbf{u}_{ub}]$

2. Define $\boldsymbol{\phi}_t(\mathbf{u}_{1:T-1}, \mathbf{x}_1)$ for $t = 1, ..., T$:
    1. If $t = 1$:
        1. Return $\mathbf{x}_1$
    2. Else:
        1. $\mathbf{x} \leftarrow \mathbf{x}_1$
        2. For $i = 1$ to $t-1$:
            1. $\mathbf{x} \leftarrow \mathbf{f}_{i}(\mathbf{x}, \mathbf{u}_{i})$
        3. Return $\mathbf{x}$

3. Define objective function $J(\mathbf{u}_{1:T-1})$:
    1. $J \leftarrow c_T(\boldsymbol{\phi}_T(\mathbf{u}_{1:T-1}, \mathbf{x}_1))$
    2. For $t = 1$ to $T-1$:
        1. $J \leftarrow J + c_t(\boldsymbol{\phi}_t(\mathbf{u}_{1:T-1}, \mathbf{x}_1), \mathbf{u}_t)$
    3. Return $J$

4. Solve optimization problem:
   $\mathbf{u}^*_{1:T-1} \leftarrow \arg\min_{\mathbf{u}_{1:T-1}} J(\mathbf{u}_{1:T-1})$
   subject to $\mathbf{u}_{lb} \leq \mathbf{u}_t \leq \mathbf{u}_{ub}, \, t=1,\ldots,T-1$

5. Return $\mathbf{u}^*_{1:T-1}$
```

```{code-cell} ipython3
:label: naive-single-shooting-impl
:tags: [hide-cell]
:load: code/naive_single_shooting.py
```

The approach outlined in {prf:ref}`naive-single-shooting` stems directly from the mathematical definition and involves recomputing the sequence of states from the begining every time that the instantenous cost function along the trajectory needs to be evaluated. This implementation has the benefit that it requires very little storage, as the only quantity that we have to maintain in addition to the running cost is the last state. However, this simplicitity and storage savings come at a steep computation cost as it requires re-computing the trajectory up to any given stage starting from the initial state. 
A more practical and efficient implementation combines trajectory unrolling with cost accumulation. This process can be realized through a simple for-loop in frameworks like JAX, which can trace code execution through control flows. Alternatively, a more efficient `scan` operation could be employed. By simultaneously computing the trajectory and summing costs, we eliminate redundant calculations, effectively trading computation for storage—a strategy reminiscent of checkpointing in automatic differentiation.

```{prf:algorithm} Single Shooting: Trajectory Storage
:label: shooting-trajectory-storage

**Inputs** Initial state $\mathbf{x}_1$, time horizon $T$, control bounds $\mathbf{u}_{lb}$ and $\mathbf{u}_{ub}$, state transition functions $\mathbf{f}_t$, cost functions $c_t$

**Output** Optimal control sequence $\mathbf{u}^*_{1:T-1}$

1. Initialize $\mathbf{u}_{1:T-1}$ within bounds $[\mathbf{u}_{lb}, \mathbf{u}_{ub}]$

2. Define function ComputeTrajectoryAndCost($\mathbf{u}_{1:T-1}, \mathbf{x}_1$):
    1. Initialize $\mathbf{x} \leftarrow [\mathbf{x}_1]$  // List to store states
    2. Initialize $J \leftarrow 0$  // Total cost
    3. For $t = 1$ to $T-1$:
        1. $J \leftarrow J + c_t(\mathbf{x}[t], \mathbf{u}_t)$
        2. $\mathbf{x}_{\text{next}} \leftarrow \mathbf{f}_t(\mathbf{x}[t], \mathbf{u}_t)$
        3. Append $\mathbf{x}_{\text{next}}$ to $\mathbf{x}$
    4. $J \leftarrow J + c_T(\mathbf{x}[T])$  // Add final state cost
    5. Return $\mathbf{x}, J$

3. Define objective function $J(\mathbf{u}_{1:T-1})$:
    1. $\_, J \leftarrow$ ComputeTrajectoryAndCost($\mathbf{u}_{1:T-1}, \mathbf{x}_1$)
    2. Return $J$

4. Solve optimization problem:
   $\mathbf{u}^*_{1:T-1} \leftarrow \arg\min_{\mathbf{u}_{1:T-1}} J(\mathbf{u}_{1:T-1})$
   subject to $\mathbf{u}_{lb} \leq \mathbf{u}_t \leq \mathbf{u}_{ub}, \, t=1,\ldots,T-1$

5. Return $\mathbf{u}^*_{1:T-1}$
```

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show code demonstration"
:  code_prompt_hide: "Hide code demonstration"
:load: code/single_shooting_unrolled.py
```

### Dealing with Bound Constraints

While we have successfully eliminated the dynamics as explicit constraints through what essentially amounts to a "reparametrization" of our problem, we've been silent regarding the bound constraints. The view of single shooting as a perfect transformation from a constrained problem to an unconstrained one is not entirely accurate: we must leave something on the table, and that something is the ability to easily impose state constraints.

By directly simulating the process from the initial state, there is one and only one corresponding induced path, and there's no way to let our optimizer know that it can adjust within some bounds, even if that means the generated trajectory is no longer feasible (realistic).

Fortunately, the situation is much better for bound constraints on the controls. If we choose gradient descent as our method for solving this problem, we can consider a simple extension to readily support these kinds of bound constraints. The approach, in this case, would be what we call projected gradient descent. The general form of a projected gradient descent step can be expressed as:

$$
\mathbf{u}_{k+1} = \mathcal{P}_C(\mathbf{u}_k - \alpha \nabla J(\mathbf{u}_k))
$$

where $\mathcal{P}_C$ denotes the projection onto the feasible set $C$, $\alpha$ is the step size, and $\nabla J(\mathbf{u}_k)$ is the gradient of the objective function at the current point $\mathbf{u}_k$. In general, the projection operation can be computationally expensive or even intractable. However, in the case of box constraints (i.e., bound constraints), the projection simplifies to an element-wise clipping operation:

$$
[\mathcal{P}_C(\mathbf{u})]_i = \begin{cases}
    [\mathbf{u}_{lb}]_i & \text{if } [\mathbf{u}]_i < [\mathbf{u}_{lb}]_i \\
    [\mathbf{u}]_i & \text{if } [\mathbf{u}_{lb}]_i \leq [\mathbf{u}]_i \leq [\mathbf{u}_{ub}]_i \\
    [\mathbf{u}_{ub}]_i & \text{if } [\mathbf{u}]_i > [\mathbf{u}_{ub}]_i
\end{cases}
$$

With this simple change, we can maintain the computational simplicity of unconstrained optimization while enforcing the bound constraints at each iteration: ie ensuring that we are feasible throughout optimization. Moreover, it can be shown that this projection preserves the convergence properties of the gradient descent method, and that under suitable conditions (such as Lipschitz continuity of the gradient), projected gradient descent converges to a stationary point of the constrained problem. 

Here's the algorithm for projected gradient descent with bound constraint for a general problem of the form:

$$
\begin{align*}
\min_{\mathbf{u}} \quad & J(\mathbf{u}) \\
\text{subject to} \quad & \mathbf{u}_{lb} \leq \mathbf{u} \leq \mathbf{u}_{ub}
\end{align*}
$$

where $J(\mathbf{u})$ is our objective function, and $\mathbf{u}_{lb}$ and $\mathbf{u}_{ub}$ are the lower and upper bounds on the control variables, respectively.


```{prf:algorithm} Projected Gradient Descent for Bound Constraints
:label: proj-grad-descent-bound-constraints

**Input:** Initial point $\mathbf{u}_0$, learning rate $\alpha$, bounds $\mathbf{u}_{lb}$ and $\mathbf{u}_{ub}$, 
           maximum iterations $\max_\text{iter}$, tolerance $\varepsilon$

1. Initialize $k = 0$
2. While $k < \max_\text{iter}$ and not converged:
    1. Compute gradient: $\mathbf{g}_k = \nabla J(\mathbf{u}_k)$
    2. Update: $\mathbf{u}_{k+1} = \text{clip}(\mathbf{u}_k - \alpha \mathbf{g}_k, \mathbf{u}_{lb}, \mathbf{u}_{ub})$
    3. Check convergence: if $\|\mathbf{u}_{k+1} - \mathbf{u}_k\| < \varepsilon$, mark as converged
    4. $k = k + 1$
3. Return $\mathbf{u}_k$
```

In this algorithm, the `clip` function projects the updated point back onto the feasible region defined by the bounds:

$$
\text{clip}(u, u_{lb}, u_{ub}) = \max(\min(u, u_{ub}), u_{lb})
$$


### On the choice of optimizer

Despite frequent mentions of automatic differentiation, it's important to note that the single shooting approaches outlined in this section need not rely on gradient-based optimization methods. In fact, one could use any method provided by `scipy.optimize.minimize`, which offers a range of options such as:

- Derivative-free methods like [Nelder-Mead Simplex](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html), suitable for problems where gradients are unavailable or difficult to compute.
- Quasi-Newton methods like [BFGS (Broyden-Fletcher-Goldfarb-Shanno)](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html), which by default uses finite differences rather than automatic differentiation to approximate gradients.

Another common strategy for single shooting methods is to use stochastic optimization techniques. For instance, random search generates a number of candidate solutions randomly and evaluates them. This approach is useful for problems with badly behaved loss landscapes or when gradient information is unreliable. More sophisticated stochastic methods include:

- Genetic Algorithms: These mimic biological evolution, using mechanisms like selection, crossover, and mutation to evolve a population of solutions over generations {cite}`holland1992genetic`. (Implemented in [DEAP](https://github.com/DEAP/deap) library)
- Simulated Annealing: Inspired by the annealing process in metallurgy, this method allows for occasional "uphill" moves to escape local minima {cite}`kirkpatrick1983optimization`. (Available in [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html))
- Particle Swarm Optimization: This technique simulates the social behavior of organisms in a swarm, with particles (candidate solutions) moving through the search space and influencing each other {cite}`kennedy1995particle`. (Implemented in [PySwarms](https://github.com/ljvmiranda921/pyswarms) library)

The selection of an optimization method for single shooting is influenced by multiple factors: problem-specific characteristics, available computational resources, and the balance between exploring the solution space and exploiting known good solutions. While gradient-based methods generally offer faster convergence when applicable, derivative-free and stochastic approaches tend to be more robust to complex non-convex loss landscapes, albeit at the cost of increased computational demands.

In practice, however, this choice is often guided by the tools at hand and the practitioners' familiarity with them. For instance, researchers with a background in deep learning tend to gravitate towards first-order gradient-based optimization techniques along with automatic differentiation for efficient derivative computation.  

## Constrained Optimization Approach

The mathematical programming formulation presented earlier lends itself readily to off-the-shelf solvers for nonlinear mathematical programs. For example, we can use the `scipy.optimize.minimize` function along with the SLSQP (Sequential Least Squares Programming) solver to obtain a solution to any feasible Bolza problem of the form presented below.

The following code demonstrate how the car charging problem can be solved directly using `scipy.optimize.minimize` in a black box fashion. In the next sections, we will dive deeper into the mathematical underpinnings of constrained optimization and implement our own solvers. For the moment, I simply want to bring to your attention that the solution to this problem, as expressed in its original form, involves solving for two interdependent quantities: the optimal sequence of controls, and that of the states encountered when applying them to the system. Is that bug, or a feature? We'll see that it depends...

```{code-cell} ipython3
:tags: [hide-input]
:load: code/example_docp.py
```

### Nonlinear Programming

Unless specific assumptions are made on the dynamics and cost structure, a DOCP is, in its most general form, a nonlinear mathematical program (commonly referred to as an NLP, not to be confused with Natural Language Processing). An NLP can be formulated as follows:

$$
\begin{aligned}
\text{minimize } & f(\mathbf{x}) \\
\text{subject to } & \mathbf{g}(\mathbf{x}) \leq \mathbf{0} \\
& \mathbf{h}(\mathbf{x}) = \mathbf{0}
\end{aligned}
$$

Where:
- $f: \mathbb{R}^n \to \mathbb{R}$ is the objective function
- $\mathbf{g}: \mathbb{R}^n \to \mathbb{R}^m$ represents inequality constraints
- $\mathbf{h}: \mathbb{R}^n \to \mathbb{R}^\ell$ represents equality constraints

Unlike unconstrained optimization commonly used in deep learning, the optimality of a solution in constrained optimization must consider both the objective value and constraint feasibility. To illustrate this, consider the following problem, which includes both equality and inequality constraints:

$$
\begin{align*}
\text{Minimize} \quad & f(x_1, x_2) = (x_1 - 1)^2 + (x_2 - 2.5)^2 \\
\text{subject to} \quad & g(x_1, x_2) = (x_1 - 1)^2 + (x_2 - 1)^2 \leq 1.5, \\
& h(x_1, x_2) = x_2 - \left(0.5 \sin(2 \pi x_1) + 1.5\right) = 0.
\end{align*}
$$

In this example, the objective function $f(x_1, x_2)$ is quadratic, the inequality constraint $g(x_1, x_2)$ defines a circular feasible region centered at $(1, 1)$ with a radius of $\sqrt{1.5}$ and the equality constraint $h(x_1, x_2)$ requires $x_2$ to lie on a sine wave function. The following code demonstrates the difference between the unconstrained, and constrained solutions to this problem. 

```{code-cell} ipython3
:tags: [hide-input]
:load: code/nlp_geometry.py
```

#### Karush-Kuhn-Tucker (KKT) conditions

While this example is simple enough to convince ourselves visually of the solution to this particular problem, it falls short of providing us with actionable chracterization of what constitutes and optimal solution in general. 
The Karush-Kuhn-Tucker (KKT) conditions provide us with an answer to this problem by generalizing the first-order optimality conditions in unconstrained optimization to problems involving both equality and inequality constraints.
This result relies on the construction of an auxiliary function called the Lagrangian, defined as: 

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\mu}, \boldsymbol{\lambda})=f(\mathbf{x})+\boldsymbol{\mu}^{\top} \mathbf{g}(\mathbf{x})+\boldsymbol{\lambda}^{\top} \mathbf{h}(\mathbf{x})$$

where $\boldsymbol{\mu} \in \mathbb{R}^m$ and $\boldsymbol{\lambda} \in \mathbb{R}^\ell$ are known as Lagrange multipliers. The first-order optimality conditions then state that if $\mathbf{x}^*$, then there must exist corresponding Lagrange multipliers $\boldsymbol{\mu}^*$ and $\boldsymbol{\lambda}^*$ such that: 

````{prf:definition}
:label: kkt-conditions
1. The gradient of the Lagrangian with respect to $\mathbf{x}$ must be zero at the optimal point (**stationarity**):

   $$\nabla_x \mathcal{L}(\mathbf{x}^*, \boldsymbol{\mu}^*, \boldsymbol{\lambda}^*) = \nabla f(\mathbf{x}^*) + \sum_{i=1}^m \mu_i^* \nabla g_i(\mathbf{x}^*) + \sum_{j=1}^\ell \lambda_j^* \nabla h_j(\mathbf{x}^*) = \mathbf{0}$$

   In the case where we only have equality constraints, this means that the gradient of the objective and that of constraint are parallel to each other at the optimum but point in opposite directions. 

2. A valid solution of a NLP is one which satisfies all the constraints (**primal feasibility**)

   $$\begin{aligned}
   \mathbf{g}(\mathbf{x}^*) &\leq \mathbf{0}, \enspace \text{and} \enspace \mathbf{h}(\mathbf{x}^*) &= \mathbf{0}
   \end{aligned}$$

3. Furthermore, the Lagrange multipliers for **inequality** constraints must be non-negative (**dual feasibility**)

   $$\boldsymbol{\mu}^* \geq \mathbf{0}$$

   This condition stems from the fact that the inequality constraints can only push the solution in one direction.

4. Finally, for each inequality constraint, either the constraint is active (equality holds) or its corresponding Lagrange multiplier is zero at an optimal solution (**complementary slackness**)

   $$\mu_i^* g_i(\mathbf{x}^*) = 0, \quad \forall i = 1,\ldots,m$$
````


Let's now solve our example problem above, this time using [Ipopt](https://coin-or.github.io/Ipopt/) via the [Pyomo](http://www.pyomo.org/) interface so that we can access the Lagrange multipliers found by the solver.

```{code-cell} ipython3
:tags: [hide-cell]
:load: code/kkt_lagrangian_verif.py
```
After running the code, we find that the Lagrange multiplier associated with the inequality constraint is approximately {glue:text}`ineq_constraint[None]:.2e`. This very small value, close to zero, suggests that the inequality constraint is not active at the optimal solution, meaning that the solution point lies inside the circle defined by this constraint. This can be verified visually in the figure above. As for the equality constraint, its corresponding Lagrange multiplier is {glue:text}`eq_constraint[None]:.2e` and the fact that it's non-zero indicates that this constraint is active at the optimal solution. In general when we find a Lagrange multiplier close to zero (like the one for the inequality constraint), it means that constraint is not "binding"—the optimal solution does not lie on the boundary defined by this constraint. In contrast, a non-zero Lagrange multiplier, such as the one for the equality constraint, indicates that the constraint is active and that any relaxation would directly affect the objective function's value, as required by the stationarity condition.

#### Lagrange Multiplier Theorem

The KKT conditions introduced above characterize the solution structure of constrained optimization problems with equality constraints. In this particular context, these conditions are referred to as the first-order optimality conditions, as part of the Lagrange multiplier theorem. Let's just re-state them in that simpler setting:

````{prf:definition} Lagrange Multiplier Theorem
Consider the constrained optimization problem:

$$
\begin{aligned}
\min_{\mathbf{x}} \quad & f(\mathbf{x}) \\
\text{subject to} \quad & h_i(\mathbf{x}) = 0, \quad i = 1, \ldots, m
\end{aligned}
$$

where $\mathbf{x} \in \mathbb{R}^n$, $f: \mathbb{R}^n \to \mathbb{R}$, and $h_i: \mathbb{R}^n \to \mathbb{R}$ for $i = 1, \ldots, m$.

Assume that:
1. $f$ and $h_i$ are continuously differentiable functions.
2. The gradients $\nabla h_i(\mathbf{x}^*)$ are linearly independent at the optimal point $\mathbf{x}^*$.

Then, there exist unique Lagrange multipliers $\lambda_i^* \in \mathbb{R}$, $i = 1, \ldots, m$, such that the following first-order optimality conditions hold:

1. Stationarity: $\nabla f(\mathbf{x}^*) + \sum_{i=1}^m \lambda_i^* \nabla h_i(\mathbf{x}^*) = \mathbf{0}$
2. Primal feasibility: $h_i(\mathbf{x}^*) = 0$, for $i = 1, \ldots, m$
````

Note that both the stationarity and primal feasibility statements are simply saying that the derivative of the Lagrangian in either the primal or dual variables must be zero at an optimal constrained solution. In other words:

$$
\nabla_{\mathbf{x}, \boldsymbol{\lambda}} L(\mathbf{x}^*, \boldsymbol{\lambda}^*) = \mathbf{0}
$$

Letting $\mathbf{F}(\mathbf{x}, \boldsymbol{\lambda})$ stand for $\nabla_{\mathbf{x}, \boldsymbol{\lambda}} L(\mathbf{x}, \boldsymbol{\lambda})$, the Lagrange multipliers theorem tells us that an optimal primal-dual pair is actually a zero of that function $\mathbf{F}$: the derivative of the Lagrangian. Therefore, we can use this observation to craft a solution method for solving equality constrained optimization using Newton's method, which is a numerical procedure for finding zeros of a nonlinear function.

#### Newton's Method

Newton's method is a numerical procedure for solving root-finding problems. These are nonlinear systems of equations of the form:

Find $\mathbf{z}^* \in \mathbb{R}^n$ such that $\mathbf{F}(\mathbf{z}^*) = \mathbf{0}$

where $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^n$ is a continuously differentiable function. Newton's method then consists in applying the following sequence of iterates:

$$
\mathbf{z}^{k+1} = \mathbf{z}^k - [\nabla \mathbf{F}(\mathbf{z}^k)]^{-1} \mathbf{F}(\mathbf{z}^k)
$$

where $\mathbf{z}^k$ is the k-th iterate, and $\nabla \mathbf{F}(\mathbf{z}^k)$ is the Jacobian matrix of $\mathbf{F}$ evaluated at $\mathbf{z}^k$.

Newton's method exhibits local quadratic convergence: if the initial guess $\mathbf{z}^0$ is sufficiently close to the true solution $\mathbf{z}^*$, and $\nabla \mathbf{F}(\mathbf{z}^*)$ is nonsingular, the method converges quadratically to $\mathbf{z}^*$ {cite}`ortega_rheinboldt_1970`. However, the method is sensitive to the initial guess; if it's too far from the desired solution, Newton's method might fail to converge or converge to a different root. To mitigate this problem, a set of techniques known as numerical continuation methods {cite}`allgower_georg_1990` have been developed. These methods effectively enlarge the basin of attraction of Newton's method by solving a sequence of related problems, progressing from an easy one to the target problem. This approach is reminiscent of several concepts in machine learning and statistical inference: curriculum learning in machine learning, where models are trained on increasingly complex data; tempering in Markov Chain Monte Carlo (MCMC) samplers, which gradually adjusts the target distribution to improve mixing; and modern diffusion models, which use a similar concept of gradually transforming noise into structured data.

##### Efficient Implementation of Newton's Method

Note that each step of Newton's method involves computing the inverse of a Jacobian matrix. However, a cardinal rule in numerical linear algebra is to avoid computing matrix inverses explicitly: rarely, if ever, should there be a `np.lindex.inv` in your code. Instead, the numerically stable and computationally efficient approach is to solve a linear system of equations at each step.
Given the Newton's method iterate:

$$
\mathbf{z}^{k+1} = \mathbf{z}^k - [\nabla \mathbf{F}(\mathbf{z}^k)]^{-1} \mathbf{F}(\mathbf{z}^k)
$$

We can reformulate this as a two-step procedure:

1. Solve the linear system: $\underbrace{[\nabla \mathbf{F}(\mathbf{z}^k)]}_{\mathbf{A}} \Delta \mathbf{z}^k = -\mathbf{F}(\mathbf{z}^k)$
2. Update: $\mathbf{z}^{k+1} = \mathbf{z}^k + \Delta \mathbf{z}^k$

The structure of the linear system in step 1 often allows for specialized solution methods. In the context of automatic differentiation, matrix-free linear solvers are particularly useful. These solvers can find a solution without explicitly forming the matrix A, requiring only the ability to evaluate matrix-vector or vector-matrix products. Typical examples of such methods include classical matrix-splitting methods (e.g., Richardson iteration) or conjugate gradient methods through [`sparse.linalg.cg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html) for example. Another useful method is the Generalized Minimal Residual method (GMRES) implemented in SciPy via [`sparse.linalg.gmres`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html), which is useful when facing non-symmetric and indefinite systems.

By inspecting the structure of matrix $\mathbf{A}$ in the specific application where the function $\mathbf{F}$ is the derivative of the Lagrangian, we will also uncover an important structure known as the KKT matrix. This structure will then allow us to derive a Quadratic Programming (QP) sub-problem as part of a larger iterative procedure for solving equality and inequality constrained problems via Sequential Quadratic Programming (SQP).

#### Solving Equality Constrained Programs with Newton's Method

To solve equality-constrained optimization problems using Newton's method, we begin by recognizing that the problem reduces to finding a zero of the function $\mathbf{F}(\mathbf{z}) = \nabla_{\mathbf{x}, \boldsymbol{\lambda}} L(\mathbf{x}, \boldsymbol{\lambda})$. Here, $\mathbf{F}$ represents the derivative of the Lagrangian function, and $\mathbf{z} = (\mathbf{x}, \boldsymbol{\lambda})$ combines both the primal variables $\mathbf{x}$ and the dual variables (Lagrange multipliers) $\boldsymbol{\lambda}$. Explicitly, we have:

$$
\mathbf{F}(\mathbf{z}) = \begin{bmatrix} \nabla_{\mathbf{x}} L(\mathbf{x}, \boldsymbol{\lambda}) \\ \mathbf{h}(\mathbf{x}) \end{bmatrix} = \begin{bmatrix} \nabla f(\mathbf{x}) + \sum_{i=1}^m \lambda_i \nabla h_i(\mathbf{x}) \\ \mathbf{h}(\mathbf{x}) \end{bmatrix}.
$$

Newton's method involves linearizing $\mathbf{F}(\mathbf{z})$ around the current iterate $\mathbf{z}^k = (\mathbf{x}^k, \boldsymbol{\lambda}^k)$ and then solving the resulting linear system. At each iteration $k$, Newton's method updates the current estimate by solving the linear system:

$$
\mathbf{z}^{k+1} = \mathbf{z}^k - [\nabla \mathbf{F}(\mathbf{z}^k)]^{-1} \mathbf{F}(\mathbf{z}^k).
$$

However, instead of explicitly inverting the Jacobian matrix $\nabla \mathbf{F}(\mathbf{z}^k)$, we solve the linear system:

$$
\underbrace{\nabla \mathbf{F}(\mathbf{z}^k)}_{\mathbf{A}} \Delta \mathbf{z}^k = -\mathbf{F}(\mathbf{z}^k),
$$

where $\Delta \mathbf{z}^k = (\Delta \mathbf{x}^k, \Delta \boldsymbol{\lambda}^k)$ represents the Newton step for the primal and dual variables. Substituting the expression for $\mathbf{F}(\mathbf{z})$ and its Jacobian, the system becomes:

$$
\begin{bmatrix}
\nabla^2_{\mathbf{x}\mathbf{x}} L(\mathbf{x}^k, \boldsymbol{\lambda}^k) & \nabla \mathbf{h}(\mathbf{x}^k)^T \\
\nabla \mathbf{h}(\mathbf{x}^k) & \mathbf{0}
\end{bmatrix}
\begin{bmatrix}
\Delta \mathbf{x}^k \\
\Delta \boldsymbol{\lambda}^k
\end{bmatrix}
=
-
\begin{bmatrix}
\nabla f(\mathbf{x}^k) + \nabla \mathbf{h}(\mathbf{x}^k)^T \boldsymbol{\lambda}^k \\
\mathbf{h}(\mathbf{x}^k)
\end{bmatrix}.
$$

The matrix on the left-hand side is known as the KKT matrix, as it stems from the Karush-Kuhn-Tucker conditions for this optimization problem
The solution of this system provides the updates $\Delta \mathbf{x}^k$ and $\Delta \boldsymbol{\lambda}^k$, which are then used to update the primal and dual variables:

$$
\mathbf{x}^{k+1} = \mathbf{x}^k + \Delta \mathbf{x}^k, \quad \boldsymbol{\lambda}^{k+1} = \boldsymbol{\lambda}^k + \Delta \boldsymbol{\lambda}^k.
$$

##### Demonstration
The following code demonstates how we can implement this idea in Jax. In this demonstration, we are minimizing a quadratic objective function subject to a single equality constraint, a problem formally stated as follows:

$$
\begin{aligned}
\min_{x \in \mathbb{R}^2} \quad & f(x) = (x_1 - 2)^2 + (x_2 - 1)^2 \\
\text{subject to} \quad & h(x) = x_1^2 + x_2^2 - 1 = 0
\end{aligned}
$$

Geometrically speaking, the constraint $h(x)$ describes a unit circle centered at the origin. To solve this problem using the method of Lagrange multipliers, we form the Lagrangian:

$$
L(x, \lambda) = f(x) + \lambda h(x) = (x_1 - 2)^2 + (x_2 - 1)^2 + \lambda(x_1^2 + x_2^2 - 1)
$$

For this particular problem, it happens so that we can also find an analytical without even having to use Newton's method. From the first-order optimality conditions, we obtain the following linear system of equations: 
\begin{align*}
   2(x_1 - 2) + 2\lambda x_1 &= 0 \\
   2(x_2 - 1) + 2\lambda x_2 &= 0 \\
   x_1^2 + x_2^2 - 1 &= 0\\
\end{align*}

From the first two equations, we then get:
 
   $$x_1 = \frac{2}{1 + \lambda}, \quad x_2 = \frac{1}{1 + \lambda}$$

which we can substitute these into the 3rd constraint equation to obtain:
   
   $$(\frac{2}{1 + \lambda})^2 + (\frac{1}{1 + \lambda})^2 = 1 \Leftrightarrow \lambda = \sqrt{5} - 1$$$

This value of the Lagrange multiplier can then be backsubstituted into the above equations to obtain $x_1 = \frac{2}{\sqrt{5}}$ and $x_2 =  \frac{1}{\sqrt{5}}$.
We can verify numerically (and visually on the following graph) that the point $(2/\sqrt{5}, 1/\sqrt{5})$ is indeed the point on the unit circle closest to $(2, 1)$.


```{code-cell} ipython3
:tags: [hide-input]
:load: code/ecp_newton.py
```

### The SQP Approach: Taylor Expansion and Quadratic Approximation

Sequential Quadratic Programming (SQP) tackles the problem of solving constrained programs by iteratively solving a sequence of simpler subproblems. Specifically, these subproblems are quadratic programs (QPs) that approximate the original problem around the current iterate by using a quadratic model of the objective function and a linear model of the constraints. Suppose we have the following optimization problem with equality constraints:

$$
\begin{aligned}
\min_{\mathbf{x}} \quad & f(\mathbf{x}) \\
\text{subject to} \quad & \mathbf{h}(\mathbf{x}) = \mathbf{0}.
\end{aligned}
$$

At each iteration $k$, we approximate the objective function $f(\mathbf{x})$ using a second-order Taylor expansion around the current iterate $\mathbf{x}^k$. The standard Taylor expansion for $f$ would be:

\begin{align*}
f(\mathbf{x}) \approx f(\mathbf{x}^k) + \nabla f(\mathbf{x}^k)^T (\mathbf{x} - \mathbf{x}^k) + \frac{1}{2} (\mathbf{x} - \mathbf{x}^k)^T \nabla^2 f(\mathbf{x}^k) (\mathbf{x} - \mathbf{x}^k).
\end{align*}

This expansion uses the **Hessian of the objective function** $\nabla^2 f(\mathbf{x}^k)$ to capture the curvature of $f$. However, in the context of constrained optimization, we also need to account for the effect of the constraints on the local behavior of the solution. If we were to use only $\nabla^2 f(\mathbf{x}^k)$, we would not capture the influence of the constraints on the curvature of the feasible region. The resulting subproblem might then lead to steps that violate the constraints or are less effective in achieving convergence. The choice that we make instead is to use the Hessian of the Lagrangian, $\nabla^2_{\mathbf{x}\mathbf{x}} L(\mathbf{x}^k, \boldsymbol{\lambda}^k)$, leading to the following quadratic model:

$$
f(\mathbf{x}) \approx f(\mathbf{x}^k) + \nabla f(\mathbf{x}^k)^T (\mathbf{x} - \mathbf{x}^k) + \frac{1}{2} (\mathbf{x} - \mathbf{x}^k)^T \nabla^2_{\mathbf{x}\mathbf{x}} L(\mathbf{x}^k, \boldsymbol{\lambda}^k) (\mathbf{x} - \mathbf{x}^k).
$$

Similarly, the equality constraints $\mathbf{h}(\mathbf{x})$ are linearized around $\mathbf{x}^k$:

$$
\mathbf{h}(\mathbf{x}) \approx \mathbf{h}(\mathbf{x}^k) + \nabla \mathbf{h}(\mathbf{x}^k) (\mathbf{x} - \mathbf{x}^k).
$$

Combining these approximations, we obtain a Quadratic Programming (QP) subproblem, which approximates our original problem locally at $\mathbf{x}^k$ but is easier to solve:

$$
\begin{aligned}
\text{Minimize} \quad & \nabla f(\mathbf{x}^k)^T \Delta \mathbf{x} + \frac{1}{2} \Delta \mathbf{x}^T \nabla^2_{\mathbf{x}\mathbf{x}} L(\mathbf{x}^k, \boldsymbol{\lambda}^k) \Delta \mathbf{x} \\
\text{subject to} \quad & \nabla \mathbf{h}(\mathbf{x}^k) \Delta \mathbf{x} + \mathbf{h}(\mathbf{x}^k) = \mathbf{0},
\end{aligned}
$$

where $\Delta \mathbf{x} = \mathbf{x} - \mathbf{x}^k$. The QP subproblem solved at each iteration focuses on finding the optimal step direction $\Delta \mathbf{x}$ for the primal variables.
While solving this QP, we obtain not only the step $\Delta \mathbf{x}$ but also the associated Lagrange multipliers for the QP subproblem, which correspond to an updated dual variable vector $\boldsymbol{\lambda}^{k+1}$. More specifically, after solving the QP, we use $\Delta \mathbf{x}^k$ to update the primal variables:

\begin{align*}
\mathbf{x}^{k+1} = \mathbf{x}^k + \Delta \mathbf{x}^k.
\end{align*}

Simultaneously, the Lagrange multipliers from the QP provide the updated dual variables $\boldsymbol{\lambda}^{k+1}$.
We summarize the SQP algorithm in the following pseudo-code: 

````{prf:algorithm} Sequential Quadratic Programming (SQP)
:label: alg-sqp

**Input:** Initial estimate $\mathbf{x}^0$, initial Lagrange multipliers $\boldsymbol{\lambda}^0$, tolerance $\epsilon > 0$.

**Output:** Solution $\mathbf{x}^*$, Lagrange multipliers $\boldsymbol{\lambda}^*$.

**Procedure:**

1. **Compute the QP Solution:** Solve the QP subproblem to obtain $\Delta \mathbf{x}^k$. The QP solver also provides the updated Lagrange multipliers $\boldsymbol{\lambda}^{k+1}$ associated with the constraints.

2. **Update the Estimates:** Update the primal variables:

   $$
   \mathbf{x}^{k+1} = \mathbf{x}^k + \Delta \mathbf{x}^k.
   $$

   Set the dual variables to the updated values $\boldsymbol{\lambda}^{k+1}$ from the QP solution.

3. **Repeat Until Convergence:** Continue iterating until $\|\Delta \mathbf{x}^k\| < \epsilon$ and the KKT conditions are satisfied.
````

#### Connection to Newton's Method in the Equality-Constrained Case

The QP subproblem in SQP is directly related to applying Newton's method for equality-constrained optimization. To see this, note that the KKT matrix of the QP subproblem is: 

\begin{align*}
\begin{bmatrix}
\nabla^2_{\mathbf{x}\mathbf{x}} L(\mathbf{x}^k, \boldsymbol{\lambda}^k) & \nabla \mathbf{h}(\mathbf{x}^k)^T \\
\nabla \mathbf{h}(\mathbf{x}^k) & \mathbf{0}
\end{bmatrix}
\begin{bmatrix}
\Delta \mathbf{x}^k \\
\Delta \boldsymbol{\lambda}^k
\end{bmatrix}
=
-
\begin{bmatrix}
\nabla f(\mathbf{x}^k) + \nabla \mathbf{h}(\mathbf{x}^k)^T \boldsymbol{\lambda}^k \\
\mathbf{h}(\mathbf{x}^k)
\end{bmatrix}
\end{align*}

This is exactly the same linear system that have to solve when applying Newton's method to the KKT conditions of the original program! Thus, solving the QP subproblem at each iteration of SQP is equivalent to taking a Newton step on the KKT conditions of the original nonlinear problem.

### SQP for Inequality-Constrained Optimization

So far, we've applied the ideas behind Sequential Quadratic Programming (SQP) to problems with only equality constraints. Now, let's extend this framework to handle optimization problems that also include inequality constraints.
Consider a general nonlinear optimization problem that includes both equality and inequality constraints:


\begin{align*}
\min_{\mathbf{x}} \quad & f(\mathbf{x}) \\
\text{subject to} \quad & \mathbf{g}(\mathbf{x}) \leq \mathbf{0}, \\
& \mathbf{h}(\mathbf{x}) = \mathbf{0}.
\end{align*}

As we did earlier, we approximate this problem by constructing a quadratic approximation to the objective and a linearization of the constraints. QP subproblem at each iteration is then formulated as:

\begin{align*}
\text{Minimize} \quad & \nabla f(\mathbf{x}^k)^T \Delta \mathbf{x} + \frac{1}{2} \Delta \mathbf{x}^T \nabla^2_{\mathbf{x}\mathbf{x}} L(\mathbf{x}^k, \boldsymbol{\lambda}^k, \boldsymbol{\nu}^k) \Delta \mathbf{x} \\
\text{subject to} \quad & \nabla \mathbf{g}(\mathbf{x}^k) \Delta \mathbf{x} + \mathbf{g}(\mathbf{x}^k) \leq \mathbf{0}, \\
& \nabla \mathbf{h}(\mathbf{x}^k) \Delta \mathbf{x} + \mathbf{h}(\mathbf{x}^k) = \mathbf{0},
\end{align*}

where $\Delta \mathbf{x} = \mathbf{x} - \mathbf{x}^k$ represents the step direction for the primal variables. The following pseudocode outlines the steps involved in applying SQP to a problem with both equality and inequality constraints:

````{prf:algorithm} Sequential Quadratic Programming (SQP) with Inequality Constraints
:label: alg-sqp-ineq

**Input:** Initial estimate $\mathbf{x}^0$, initial multipliers $\boldsymbol{\lambda}^0, \boldsymbol{\nu}^0$, tolerance $\epsilon > 0$.

**Output:** Solution $\mathbf{x}^*$, Lagrange multipliers $\boldsymbol{\lambda}^*, \boldsymbol{\nu}^*$.

**Procedure:**

1. **Initialization:**
   Set $k = 0$.

2. **Repeat:**

   a. **Construct the QP Subproblem:**
   Formulate the QP subproblem using the current iterate $\mathbf{x}^k$, $\boldsymbol{\lambda}^k$, and $\boldsymbol{\nu}^k$:

   $$
   \begin{aligned}
   \text{Minimize} \quad & \nabla f(\mathbf{x}^k)^T \Delta \mathbf{x} + \frac{1}{2} \Delta \mathbf{x}^T \nabla^2_{\mathbf{x}\mathbf{x}} L(\mathbf{x}^k, \boldsymbol{\lambda}^k, \boldsymbol{\nu}^k) \Delta \mathbf{x} \\
   \text{subject to} \quad & \nabla \mathbf{g}(\mathbf{x}^k) \Delta \mathbf{x} + \mathbf{g}(\mathbf{x}^k) \leq \mathbf{0}, \\
   & \nabla \mathbf{h}(\mathbf{x}^k) \Delta \mathbf{x} + \mathbf{h}(\mathbf{x}^k) = \mathbf{0}.
   \end{aligned}
   $$

   b. **Solve the QP Subproblem:**
   Solve for $\Delta \mathbf{x}^k$ and obtain the updated Lagrange multipliers $\boldsymbol{\lambda}^{k+1}$ and $\boldsymbol{\nu}^{k+1}$.

   c. **Update the Estimates:**
   Update the primal variables and multipliers:

   $$
   \mathbf{x}^{k+1} = \mathbf{x}^k + \Delta \mathbf{x}^k.
   $$

   d. **Check for Convergence:**
   If $\|\Delta \mathbf{x}^k\| < \epsilon$ and the KKT conditions are satisfied, stop. Otherwise, set $k = k + 1$ and repeat.

3. **Return:**
   $\mathbf{x}^* = \mathbf{x}^{k+1}, \boldsymbol{\lambda}^* = \boldsymbol{\lambda}^{k+1}, \boldsymbol{\nu}^* = \boldsymbol{\nu}^{k+1}$.
````

#### Demonstration with JAX and CVXPy

Consider the following equality and inequality-constrained problem:

\begin{align*}
\min_{x \in \mathbb{R}^2} \quad & f(x) = (x_1 - 2)^2 + (x_2 - 1)^2 \\
\text{subject to} \quad & g(x) = x_1^2 - x_2 \leq 0  \\
& h(x) = x_1^2 + x_2^2 - 1 = 0
\end{align*}

This example builds on our previous one but adds a parabola-shaped inequality constraint. We require our solution to lie not only on the circle defining our equality constraint but also below the parabola. To solve the QP subproblem, we will be using the [CVXPY](https://www.cvxpy.org/) package. While the Lagrangian and derivatives could be computed easily by hand, we use [JAX](https://jax.readthedocs.io/) for generality:

```{code-cell} ipython3
:tags: [hide-input]
:load: code/sqp_ineq_cvxpy_jax.py
```

### The Arrow-Hurwicz-Uzawa algorithm

While the SQP method addresses constrained optimization problems by sequentially solving quadratic subproblems, an alternative approach emerges from viewing constrained optimization as a min-max problem. This perspective leads to a simpler algorithm, originally introduced by the Arrow-Hurwicz-Uzawa {cite}`arrow1958studies`. Consider the following general constrained optimization problem encompassing both equality and inequality constraints:

$$
\begin{aligned}
\min_{\mathbf{x}} \quad & f(\mathbf{x}) \\
\text{subject to} \quad & \mathbf{g}(\mathbf{x}) \leq \mathbf{0} \\
& \mathbf{h}(\mathbf{x}) = \mathbf{0}
\end{aligned}
$$

Using the Lagrangian function $L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\mu}) = f(\mathbf{x}) + \boldsymbol{\mu}^T \mathbf{g}(\mathbf{x}) + \boldsymbol{\lambda}^T \mathbf{h}(\mathbf{x})$, we can reformulate this problem as the following min-max problem:

$$
\min_{\mathbf{x}} \max_{\boldsymbol{\lambda}, \boldsymbol{\mu} \geq 0} L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\mu})
$$

The role of each component in this min-max structure can be understood as follows:

1. The outer minimization over $\mathbf{x}$ finds the feasible point that minimizes the objective function $f(\mathbf{x})$.
2. The maximization over $\boldsymbol{\mu} \geq 0$ ensures that inequality constraints $\mathbf{g}(\mathbf{x}) \leq \mathbf{0}$ are satisfied. If any inequality constraint is violated, the corresponding term in $\boldsymbol{\mu}^T \mathbf{g}(\mathbf{x})$ can be made arbitrarily large by choosing a large enough $\mu_i$.
3. The maximization over $\boldsymbol{\lambda}$ ensures that equality constraints $\mathbf{h}(\mathbf{x}) = \mathbf{0}$ are satisfied. 

Using this observation, we can devise an algorithm which, like SQP, will update both the primal and dual variables at every step. But rather than using second-order optimization, we will simply use a first-order gradient update step: a descent step in the primal variable, and an ascent step in the dual one. The corresponding procedure, when implemented by gradient descent, is called Gradient Ascent Descent in the learning and optimization communities. In the case of equality constraints only, the algorithm looks like the following:

````{prf:algorithm} Arrow-Hurwicz-Uzawa for equality constraints only
:label: ahuz-eq

**Input:** Initial guess $\mathbf{x}^0$, $\boldsymbol{\lambda}^0$, step sizes $\alpha$, $\beta$
**Output:** Optimal $\mathbf{x}^*$, $\boldsymbol{\lambda}^*$

1: **for** $k = 0, 1, 2, \ldots$ until convergence **do**

2:     $\mathbf{x}^{k+1} = \mathbf{x}^k - \alpha \nabla_{\mathbf{x}} L(\mathbf{x}^k, \boldsymbol{\lambda}^k)$  **(Primal update)**

3:     $\boldsymbol{\lambda}^{k+1} = \boldsymbol{\lambda}^k + \beta \nabla_{\boldsymbol{\lambda}} L(\mathbf{x}^{k+1}, \boldsymbol{\lambda}^k)$  **(Dual update)**

4: **end for**

5: **return** $\mathbf{x}^k$, $\boldsymbol{\lambda}^k$
````

Now to account for the fact that the Lagrange multiplier needs to be non-negative for inequality constraints, we can use our previous idea from projected gradient descent for bound constraints and consider a projection, or clipping step to ensure that this condition is satisfied throughout. In this case, the algorithm looks like the following:

````{prf:algorithm} Arrow-Hurwicz-Uzawa for equality and inequality constraints
:label: ahuz-full

**Input:** Initial guess $\mathbf{x}^0$, $\boldsymbol{\lambda}^0$, $\boldsymbol{\mu}^0 \geq 0$, step sizes $\alpha$, $\beta$, $\gamma$
**Output:** Optimal $\mathbf{x}^*$, $\boldsymbol{\lambda}^*$, $\boldsymbol{\mu}^*$

1: **for** $k = 0, 1, 2, \ldots$ until convergence **do**

2:     $\mathbf{x}^{k+1} = \mathbf{x}^k - \alpha \nabla_{\mathbf{x}} L(\mathbf{x}^k, \boldsymbol{\lambda}^k, \boldsymbol{\mu}^k)$  **(Primal update)**

3:     $\boldsymbol{\lambda}^{k+1} = \boldsymbol{\lambda}^k + \beta \nabla_{\boldsymbol{\lambda}} L(\mathbf{x}^{k+1}, \boldsymbol{\lambda}^k, \boldsymbol{\mu}^k)$  **(Dual update for equality constraints)**

4:     $\boldsymbol{\mu}^{k+1} = [\boldsymbol{\mu}^k + \gamma \nabla_{\boldsymbol{\mu}} L(\mathbf{x}^{k+1}, \boldsymbol{\lambda}^k, \boldsymbol{\mu}^k)]_+$  **(Dual update with clipping for inequality constraints)**

5: **end for**

6: **return** $\mathbf{x}^k$, $\boldsymbol{\lambda}^k$, $\boldsymbol{\mu}^k$
````

Here, $[\cdot]_+$ denotes the projection onto the non-negative orthant, ensuring that $\boldsymbol{\mu}$ remains non-negative.

However, as it is widely known from the lessons of GAN (Generative Adversarial Network) training {cite}`goodfellow2014generative`, Gradient Descent Ascent (GDA) can fail to converge or suffer from instability. The Arrow-Hurwicz-Uzawa algorithm, also known as the first-order Lagrangian method, is known to converge only locally, in the vicinity of an optimal primal-dual pair.

```{code-cell} ipython3
:tags: [hide-input]
:load: code/arrow_hurwicz_uzawa_jax.py
```

# The Discrete-Time Pontryagin Maximum Principle

Discrete-time optimal control problems (DOCPs) form a specific class of nonlinear programming problems. Therefore, we can apply the general results from the Karush-Kuhn-Tucker (KKT) conditions to characterize the structure of optimal solutions to DOCPs in any of their three forms. The discrete-time analogue of the KKT conditions for DOCPs is known as the discrete-time Pontryagin Maximum Principle (PMP). The PMP was first described by Pontryagin in 1956 {cite}`pontryagin1962mathematical` for continuous-time systems, with the discrete-time version following shortly after. Similar to the KKT conditions, the PMP is useful from both theoretical and practical perspectives. It not only allows us to sometimes find closed-form solutions but also inspires the development of algorithms.

Importantly, the PMP goes beyond the KKT conditions by demonstrating the existence of a particular recursive equation—the adjoint equation. This equation governs the evolution of the derivative of the Hamiltonian, a close cousin to the Lagrangian. The adjoint equation enables us to transform the PMP into an algorithmic procedure, which has much in common with backpropagation {cite}`rumelhart1986learning` in deep learning. This connection between optimal control theory has been noted by several researchers, including Griewank {cite}`griewank1989automatic` in the context of automatic differentiation, and LeCun {cite}`lecun1988theoretical` in his early work on neural networks.

## PMP for Mayer Problems 

Before delving into more general cases, let's consider a Mayer problem where the goal is to minimize a terminal cost function $c_T(\mathbf{x}_T)$:

$$
\begin{alignat*}{2}
\text{minimize} \quad & c_T(\mathbf{x}_T) & \\
\text{such that} \quad 
& \mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t), & \quad & t = 1, \dots, T-1, \\
& \mathbf{u}_{lb} \leq \mathbf{u}_t \leq \mathbf{u}_{ub}, & \quad & t = 1, \dots, T, \\
& \mathbf{x}_{lb} \leq \mathbf{x}_t \leq \mathbf{x}_{ub}, & \quad & t = 1, \dots, T, \\
\text{given} \quad & \mathbf{x}_1. &
\end{alignat*}
$$

As done previously using the single shooting method, we reformulate this problem as an unconstrained optimization problem (excluding the state bound constraints since we lack a straightforward way to incorporate them directly). This reformulation is:

\begin{align*}
J(\mathbf{u}_{1:T-1}) = c_T(\boldsymbol{\phi}_T(\mathbf{u}_{1:T-1}, \mathbf{x}_1)),
\end{align*}
where the state evolution functions $\boldsymbol{\phi}_t$ are defined recursively as:

\begin{align*}
\boldsymbol{\phi}_t(\mathbf{u}_{1:T-1}, \mathbf{x}_1) = 
\begin{cases}
\mathbf{x}_1, & \text{if } t = 1, \\
\mathbf{f}_{t-1}(\boldsymbol{\phi}_{t-1}(\mathbf{u}_{1:T-1}, \mathbf{x}_1), \mathbf{u}_{t-1}), & \text{if } t = 2, \ldots, T.
\end{cases}
\end{align*}

To find the first-order optimality condition, we differentiate the objective function $J(\mathbf{u}_{1:T-1})$ with respect to each control variable $\mathbf{u}_t$ and set it to zero:

$$
\frac{\partial J(\mathbf{u}_{1:T-1})}{\partial \mathbf{u}_t} = \frac{\partial c_T(\boldsymbol{\phi}_T)}{\partial \mathbf{u}_t} = 0, \quad t = 1, \ldots, T-1.
$$

Applying the chain rule, we get:

$$
\frac{\partial c_T(\boldsymbol{\phi}_T)}{\partial \mathbf{u}_t} = \frac{\partial c_T(\boldsymbol{\phi}_T)}{\partial \boldsymbol{\phi}_T} \frac{\partial \boldsymbol{\phi}_T}{\partial \mathbf{u}_t}.
$$

Now, let's expand the derivative $\frac{\partial \boldsymbol{\phi}_T}{\partial \mathbf{u}_t}$ using its non-recursive form. From the definition of the state evolution functions, we have:

\begin{align*}
\boldsymbol{\phi}_T = \mathbf{f}_{T-1}(\boldsymbol{\phi}_{T-1}, \mathbf{u}_{T-1}), \quad \boldsymbol{\phi}_{T-1} = \mathbf{f}_{T-2}(\boldsymbol{\phi}_{T-2}, \mathbf{u}_{T-2}), \quad \ldots, \quad \boldsymbol{\phi}_{t+1} = \mathbf{f}_t(\boldsymbol{\phi}_t, \mathbf{u}_t).
\end{align*}

The above can also be written more recursively. For $s \geq t$, the derivative of $\boldsymbol{\phi}_s$ with respect to $\mathbf{u}_t$ is:

$$
\frac{\partial \boldsymbol{\phi}_s}{\partial \mathbf{u}_t} = \frac{\partial \mathbf{f}_{s-1}}{\partial \boldsymbol{\phi}_{s-1}} \frac{\partial \boldsymbol{\phi}_{s-1}}{\partial \mathbf{u}_t}, \quad s = t+1, \ldots, T,
$$

and

$$
\frac{\partial \boldsymbol{\phi}_t}{\partial \mathbf{u}_t} = \frac{\partial \mathbf{f}_{t-1}}{\partial \mathbf{u}_t}.
$$

The overall derivative is then of the form:
\begin{align*}
\frac{\partial J(\mathbf{u}_{1:T-1})}{\partial \mathbf{u}_t} = \underbrace{\underbrace{\underbrace{\frac{\partial c_T(\boldsymbol{\phi}_T)}{\partial \boldsymbol{\phi}_T}}_{\boldsymbol{\lambda}_T} \frac{\partial \mathbf{f}_{T-1}}{\partial \boldsymbol{\phi}_{T-1}}}_{\boldsymbol{\lambda}_{T-1}} \cdots \frac{\partial \mathbf{f}_{t+1}}{\partial \boldsymbol{\phi}_{t+1}}}_{\boldsymbol{\lambda}_{t+1}} \frac{\partial \mathbf{f}_t}{\partial \mathbf{u}_t}.
\end{align*}
where $\boldsymbol{\lambda}_t$ is called the adjoint (co-state) variable, and contains the reverse accumulation of the derivative. The evolution of this variable also obeys a difference equation, but one which runs backward in time: the adjoint equation. The recursive relationship for the adjoint equation is then: 

$$
\boldsymbol{\lambda}_t = \frac{\partial \mathbf{f}_t}{\partial \boldsymbol{\phi}_t}^\top \boldsymbol{\lambda}_{t+1}, \quad t = 1, \ldots, T-1,
$$

with the terminal condition:

$$
\boldsymbol{\lambda}_T = \frac{\partial c_T}{\partial \boldsymbol{\phi}_T}.
$$

The first-order optimality condition in terms of the adjoint variable can finally be written as:

$$
\frac{\partial J(\mathbf{u}_{1:T-1})}{\partial \mathbf{u}_t} = \frac{\partial \mathbf{f}_t}{\partial \mathbf{u}_t}^\top \boldsymbol{\lambda}_{t+1} = 0, \quad t = 1, \ldots, T-1.
$$

## PMP for Bolza Problems

To derive the adjoint equation for the Bolza problem, we consider the optimal control problem where the objective is to minimize both a terminal cost $c_T(\mathbf{x}_T)$ and the sum of intermediate costs $c_t(\mathbf{x}_t, \mathbf{u}_t)$:

\begin{align*}
\text{minimize} \quad & c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t, \mathbf{u}_t) \\
\text{such that} \quad 
& \mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t), \quad t = 1, \dots, T-1, \\
\text{given} \quad & \mathbf{x}_1.
\end{align*}

To handle the constraints, we introduce the Lagrangian function with multipliers $\boldsymbol{\lambda}_t$ for each constraint $\mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t)$:

\begin{align*}
L(\mathbf{x}, \mathbf{u}, \boldsymbol{\lambda}) &\triangleq c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t, \mathbf{u}_t) + \sum_{t=1}^{T-1} \boldsymbol{\lambda}_{t+1}^\top \left( \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t) - \mathbf{x}_{t+1} \right).
\end{align*}

The existence of an optimal constrained solution $(\mathbf{x}^\star, \mathbf{u})^\star$ implies that there exists a unique set of Lagrange multipliers $\boldsymbol{\lambda}_t^\star$ such that the derivative of the Lagrangian with respect to all variables equals zero: $\nabla L(\mathbf{x}^\star, \mathbf{u}^\star, \boldsymbol{\lambda}^\star) = 0.$

To simplify, we rearrange the Lagrangian so that each state variable $\mathbf{x}_t$ appears only once in the summation:

\begin{align*}
L(\mathbf{x}, \mathbf{u}, \boldsymbol{\lambda}) &= c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} \left( c_t(\mathbf{x}_t, \mathbf{u}_t) + \boldsymbol{\lambda}_{t+1}^\top (\mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t) - \mathbf{x}_{t+1}) \right). \\
&= c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t, \mathbf{u}_t) + \sum_{t=1}^{T-1} \boldsymbol{\lambda}_{t+1}^\top \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t) - \sum_{t=1}^{T-1} \boldsymbol{\lambda}_{t+1}^\top \mathbf{x}_{t+1}.
\end{align*}

Note that by adding and subtracting, we can write:

$$
\sum_{t=1}^{T-1} \boldsymbol{\lambda}_{t+1}^\top \mathbf{x}_{t+1} = \boldsymbol{\lambda}_T^\top \mathbf{x}_T - \boldsymbol{\lambda}_1^\top \mathbf{x}_1 + \sum_{t=1}^{T-1} \boldsymbol{\lambda}_t^\top \mathbf{x}_t.
$$

Substituting this back into the Lagrangian gives:

\begin{align*}
L(\mathbf{x}, \mathbf{u}, \boldsymbol{\lambda}) &= c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t, \mathbf{u}_t) + \sum_{t=1}^{T-1} \boldsymbol{\lambda}_{t+1}^\top \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t) - \left( \boldsymbol{\lambda}_T^\top \mathbf{x}_T - \boldsymbol{\lambda}_1^\top \mathbf{x}_1 + \sum_{t=1}^{T-1} \boldsymbol{\lambda}_t^\top \mathbf{x}_t \right). \\
&= c_T(\mathbf{x}_T) + \boldsymbol{\lambda}_T^\top \mathbf{x}_T - \boldsymbol{\lambda}_1^\top \mathbf{x}_1 + \sum_{t=1}^{T-1} \left( c_t(\mathbf{x}_t, \mathbf{u}_t) + \boldsymbol{\lambda}_{t+1}^\top \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t) - \boldsymbol{\lambda}_t^\top \mathbf{x}_t \right).
\end{align*}

By differentiating the Lagrangian with respect to each state $\mathbf{x}_i$, we obtain:

$$
\frac{\partial L(\mathbf{x}, \mathbf{u}, \boldsymbol{\lambda})}{\partial \mathbf{x}_i} = 
\begin{cases}
\frac{\partial c_T (\mathbf{x}_T)}{\partial \mathbf{x}_T} + \boldsymbol{\lambda}_T, & \text{if } i = T, \\
\frac{\partial c_t(\mathbf{x}_t, \mathbf{u}_t)}{\partial \mathbf{x}_t} + \boldsymbol{\lambda}_{t+1}^\top \frac{\partial \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t)}{\partial \mathbf{x}_t} - \boldsymbol{\lambda}_t, & \text{if } i = 1, \dots, T-1.
\end{cases}
$$

We finally obtain the adjoint equation by setting the above expression to zero at an optimal primal-dual pair, and re-arranging the terms:

\begin{align*}
\boldsymbol{\lambda}^*_T &= \frac{\partial c_T(\mathbf{x}^*_T)}{\partial \mathbf{x}^*_T}\\
\boldsymbol{\lambda}^*_t &= \frac{\partial c_t(\mathbf{x}^*_t, \mathbf{u}^*_t)}{\partial \mathbf{x}^*_t} + (\boldsymbol{\lambda}^*_{t+1})^\top \frac{\partial \mathbf{f}_t(\mathbf{x}^*_t, \mathbf{u}^*_t)}{\partial \mathbf{x}^*_t}, \enspace t = T-1, \dots, 1
\end{align*}

The optimality condition for the controls is obtained by differentiating the Lagrangian with respect to $\mathbf{u}^*_t$:

$$
\frac{\partial L(\mathbf{x}^*, \mathbf{u}^*, \boldsymbol{\lambda}^*)}{\partial \mathbf{u}^*_t} = \frac{\partial c_t(\mathbf{x}^*_t, \mathbf{u}^*_t)}{\partial \mathbf{u}^*_t} + (\boldsymbol{\lambda}^*_{t+1})^\top \frac{\partial \mathbf{f}_t(\mathbf{x}^*_t, \mathbf{u}^*_t)}{\partial \mathbf{u}^*_t} = 0.
$$

As expected from the general theory of constrained optimization, we finally recover the fact that the constraints must be satisfied at an optimal solution: 

$$
\frac{\partial L(\mathbf{x}^*, \mathbf{u}^*, \boldsymbol{\lambda}^*)}{\partial \boldsymbol{\lambda}^*_{t+1}} = \mathbf{f}_t(\mathbf{x}^*_t, \mathbf{u}^*_t) - \mathbf{x}^*_{t+1} = 0.
$$

## Hamiltonian Formulation

The first-order optimality condition for the Bolza problem obtained above can be expressed using the so-called Hamiltonian function:

$$
H_t(\mathbf{x}_t, \mathbf{u}_t, \boldsymbol{\lambda}_{t+1}) = c_t(\mathbf{x}_t, \mathbf{u}_t) + \boldsymbol{\lambda}_{t+1}^\top \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t).
$$

If $(\mathbf{x}^*, \mathbf{u}^*)$ is a local minimum control trajectory, then:

$$
\frac{\partial H_t(\mathbf{x}_t^*, \mathbf{u}_t^*, \boldsymbol{\lambda}_{t+1}^*)}{\partial \mathbf{u}_t} = 0, \quad t = 1, \ldots, T-1,
$$
where the adjoint variables (costate vectors) \(\boldsymbol{\lambda}_t^*\) are computed from:

$$
\boldsymbol{\lambda}_t^* = \frac{\partial H_t(\mathbf{x}_t^*, \mathbf{u}_t^*, \boldsymbol{\lambda}_{t+1}^*)}{\partial \mathbf{x}_t}, \quad t = 1, \ldots, T-1, \quad \boldsymbol{\lambda}_T^* = \frac{\partial c_T(\mathbf{x}_T^*)}{\partial \mathbf{x}_T}.
$$
