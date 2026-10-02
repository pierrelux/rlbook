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

Unlike the methods we've discussed so far, dynamic programming takes a step back and considers an entire family of related problems rather than a single optimization problem. This approach, while seemingly more complex at first glance, can often lead to efficient solutions.

Dynamic programming leverage the solution structure underlying many control problems that allows for a decomposition it into smaller, more manageable subproblems. Each subproblem is itself an optimization problem, embedded within the larger whole. This recursive structure is the foundation upon which dynamic programming constructs its solutions.

To ground our discussion, let us return to the domain of discrete-time optimal control problems (DOCPs). These problems frequently arise from the discretization of continuous-time optimal control problems. While the focus here will be on deterministic problems, these concepts extend naturally to stochastic problems by taking the expectation over the random quantities.

Consider a typical DOCP of Bolza type:

$$
\begin{align*}
\text{minimize} \quad & J \triangleq c_\mathrm{T}(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t, \mathbf{u}_t) \\
\text{subject to} \quad 
& \mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t), \quad t = 1, \ldots, T-1, \\
& \mathbf{u}_{lb} \leq \mathbf{u}_t \leq \mathbf{u}_{ub}, \quad t = 1, \ldots, T, \\
& \mathbf{x}_{lb} \leq \mathbf{x}_t \leq \mathbf{x}_{ub}, \quad t = 1, \ldots, T, \\
\text{given} \quad & \mathbf{x}_1
\end{align*}
$$

Rather than considering only the total cost from the initial time to the final time, dynamic programming introduces the concept of cost from an arbitrary point in time to the end. This leads to the definition of the "cost-to-go" or "value function" $J_k(\mathbf{x}_k)$:

$$
J_k(\mathbf{x}_k) \triangleq c_\mathrm{T}(\mathbf{x}_T) + \sum_{t=k}^{T-1} c_t(\mathbf{x}_t, \mathbf{u}_t)
$$

This function represents the total cost incurred from stage $k$ onwards to the end of the time horizon, given that the system is initialized in state $\mathbf{x}_k$ at stage $k$. Suppose the problem has been solved from stage $k+1$ to the end, yielding the optimal cost-to-go $J_{k+1}^\star(\mathbf{x}_{k+1})$ for any state $\mathbf{x}_{k+1}$ at stage $k+1$. The question then becomes: how does this information inform the decision at stage $k$?

Given knowledge of the optimal behavior from $k+1$ onwards, the task reduces to determining the optimal action $\mathbf{u}_k$ at stage $k$. This control should minimize the sum of the immediate cost $c_k(\mathbf{x}_k, \mathbf{u}_k)$ and the optimal future cost $J_{k+1}^\star(\mathbf{x}_{k+1})$, where $\mathbf{x}_{k+1}$ is the resulting state after applying action $\mathbf{u}_k$. Mathematically, this is expressed as:

$$
J_k^\star(\mathbf{x}_k) = \min_{\mathbf{u}_k} \left[ c_k(\mathbf{x}_k, \mathbf{u}_k) + J_{k+1}^\star(\mathbf{f}_k(\mathbf{x}_k, \mathbf{u}_k)) \right]
$$

This equation is known as Bellman's equation, named after Richard Bellman, who formulated the principle of optimality:

> An optimal policy has the property that whatever the previous state and decision, the remaining decisions must constitute an optimal policy with regard to the state resulting from the previous decision.

In other words, any sub-path of an optimal path, from any intermediate point to the end, must itself be optimal. This principle is the basis for the backward induction procedure which computes the optimal value function and provides closed-loop control capabilities without having to use an explicit NLP solver. 

Dynamic programming can handle nonlinear systems and non-quadratic cost functions naturally. It provides a global optimal solution, when one exists, and can incorporate state and control constraints with relative ease. However, as the dimension of the state space increases, this approach suffers from what Bellman termed the "curse of dimensionality." The computational complexity and memory requirements grow exponentially with the state dimension, rendering direct application of dynamic programming intractable for high-dimensional problems.

Fortunately, learning-based methods offer efficient tools to combat the curse of dimensionality on two fronts: by using function approximation (e.g., neural networks) to avoid explicit discretization, and by leveraging randomization through Monte Carlo methods inherent in the learning paradigm. Most of this course is dedicated to those ideas.

## Backward Recursion 

The principle of optimality provides a methodology for solving optimal control problems. Beginning at the final time horizon and working backwards, at each stage the local optimization problem given by Bellman's equation is solved. This process, termed backward recursion or backward induction, constructs the optimal value function stage by stage.

````{prf:algorithm} Backward Recursion for Dynamic Programming
:label: backward-recursion-deterministic

**Input:** Terminal cost function $c_\mathrm{T}(\cdot)$, stage cost functions $c_t(\cdot, \cdot)$, system dynamics $f_t(\cdot, \cdot)$, time horizon $\mathrm{T}$

**Output:** Optimal value functions $J_t^\star(\cdot)$ and optimal control policies $\pi_t^\star(\cdot)$ for $t = 1, \ldots, T$

1. Initialize $J_T^\star(\mathbf{x}) = c_\mathrm{T}(\mathbf{x})$ for all $\mathbf{x}$ in the state space
2. For $t = T-1, T-2, \ldots, 1$:
   1. For each state $\mathbf{x}$ in the state space:
      1. Compute $J_t^\star(\mathbf{x}) = \min_{\mathbf{u}} \left[ c_t(\mathbf{x}, \mathbf{u}) + J_{t+1}^\star(f_t(\mathbf{x}, \mathbf{u})) \right]$
      2. Compute $\pi_t^\star(\mathbf{x}) = \arg\min_{\mathbf{u}} \left[ c_t(\mathbf{x}, \mathbf{u}) + J_{t+1}^\star(f_t(\mathbf{x}, \mathbf{u})) \right]$
   2. End For
3. End For
4. Return $J_t^\star(\cdot)$, $\pi_t^\star(\cdot)$ for $t = 1, \ldots, T$
````


Upon completion of this backward pass, we now have access to the optimal control to take at any stage and in any state. Furthermore, we can simulate optimal trajectories from any initial state and applying the optimal policy at each stage to generate the optimal trajectory.



```{prf:theorem} Backward induction solves deterministic Bolza DOCP
:label: thm-bolza-backward

**Setting.** Let $\mathbf{x}_{t+1}=\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)$ for $t=1,\dots,T-1$, with admissible action sets $\mathcal{U}_t(\mathbf{x})\neq\varnothing$. Let stage costs $c_t(\mathbf{x},\mathbf{u})$ and terminal cost $c_\mathrm{T}(\mathbf{x})$ be real-valued and bounded below. Assume for every $(t,\mathbf{x})$ the one-step problem

$$
\min_{\mathbf{u}\in\mathcal{U}_t(\mathbf{x})}\big\{c_t(\mathbf{x},\mathbf{u})+J_{t+1}^\star(\mathbf{f}_t(\mathbf{x},\mathbf{u}))\big\}
$$
admits a minimizer (e.g., compact $\mathcal{U}_t(\mathbf{x})$ and continuity suffice).

Define $J_T^\star(\mathbf{x}) \equiv c_\mathrm{T}(\mathbf{x})$ and for $t=T-1,\dots,1$

$$
J_t^\star(\mathbf{x}) \;\triangleq\; \min_{\mathbf{u}\in\mathcal{U}_t(\mathbf{x})}
\Big[c_t(\mathbf{x},\mathbf{u})+J_{t+1}^\star\big(\mathbf{f}_t(\mathbf{x},\mathbf{u})\big)\Big],
$$
and select any minimizer $\boldsymbol{\pi}_t^\star(\mathbf{x})\in\arg\min(\cdot)$.

**Claim.** For every initial state $\mathbf{x}_1$, the control sequence
$\boldsymbol{\pi}_1^\star(\mathbf{x}_1),\dots,\boldsymbol{\pi}_{T-1}^\star(\mathbf{x}_{T-1})$
generated by these selectors is optimal for the Bolza problem, and
$J_1^\star(\mathbf{x}_1)$ equals the optimal cost. Moreover, $J_t^\star(\cdot)$ is the optimal cost-to-go from stage $t$ for every state, i.e., backward induction recovers the entire value function.
```

```{prf:proof}
We give a direct proof by backward induction. The general idea is that any feasible sequence can be improved by replacing its tail with an optimal continuation, so optimal solutions can be built stage by stage. This is sometimes called a "cut-and-paste" argument.

**Step 1 (verification of the recursion at a fixed stage).**  
Fix $t\in\{1,\dots,T-1\}$ and $\mathbf{x}\in\mathbb{X}$. Consider any admissible control sequence $\mathbf{u}_t,\dots,\mathbf{u}_{T-1}$ starting from $\mathbf{x}_t=\mathbf{x}$ and define the induced states $\mathbf{x}_{k+1}=\mathbf{f}_k(\mathbf{x}_k,\mathbf{u}_k)$. Its total cost from $t$ is

$$
c_t(\mathbf{x}_t,\mathbf{u}_t)+\sum_{k=t+1}^{T-1}c_k(\mathbf{x}_k,\mathbf{u}_k)+c_\mathrm{T}(\mathbf{x}_T).
$$

By definition of $J_{t+1}^\star$, the tail cost satisfies

$$
\sum_{k=t+1}^{T-1}c_k(\mathbf{x}_k,\mathbf{u}_k)+c_\mathrm{T}(\mathbf{x}_T)
\;\ge\; J_{t+1}^\star(\mathbf{x}_{t+1})
\;=\; J_{t+1}^\star\big(\mathbf{f}_t(\mathbf{x},\mathbf{u}_t)\big).
$$

Hence the total cost is bounded below by

$$
c_t(\mathbf{x},\mathbf{u}_t)+J_{t+1}^\star\big(\mathbf{f}_t(\mathbf{x},\mathbf{u}_t)\big).
$$

Taking the minimum over $\mathbf{u}_t\in\mathcal{U}_t(\mathbf{x})$ yields

$$
\text{(any admissible cost from $t$)}\;\ge\;J_t^\star(\mathbf{x}).
\tag{$\ast$}
$$

**Step 2 (existence of an optimal prefix at stage $t$).**  
By assumption, there exists $\boldsymbol{\pi}_t^\star(\mathbf{x})$ attaining the minimum in the definition of $J_t^\star(\mathbf{x})$. If we now **paste** to $\boldsymbol{\pi}_t^\star(\mathbf{x})$ an optimal tail policy from $t+1$ (whose existence we will establish inductively), the resulting sequence attains cost exactly

$$
c_t\big(\mathbf{x},\boldsymbol{\pi}_t^\star(\mathbf{x})\big)
+J_{t+1}^\star\!\Big(\mathbf{f}_t\big(\mathbf{x},\boldsymbol{\pi}_t^\star(\mathbf{x})\big)\Big)
=J_t^\star(\mathbf{x}),
$$
which matches the lower bound $(\ast)$; hence it is optimal from $t$.

**Step 3 (backward induction over time).**  
Base case $t=T$. The statement holds because $J_T^\star(\mathbf{x})=c_\mathrm{T}(\mathbf{x})$ and there is no control to choose.

Inductive step. Assume the tail statement holds for $t+1$: from any state $\mathbf{x}_{t+1}$ there exists an optimal control sequence realizing $J_{t+1}^\star(\mathbf{x}_{t+1})$. Then by Steps 1–2, selecting $\boldsymbol{\pi}_t^\star(\mathbf{x}_t)$ at stage $t$ and concatenating the optimal tail from $t+1$ yields an optimal sequence from $t$ with value $J_t^\star(\mathbf{x}_t)$.

By backward induction, the claim holds for all $t$, in particular for $t=1$ and any initial $\mathbf{x}_1$. Therefore the backward recursion both **certifies** optimality (verification) and **constructs** an optimal policy (synthesis), while recovering the full family $\{J_t^\star\}_{t=1}^T$.
```

```{prf:remark} No "big NLP" required
The Bolza DOCP over the whole horizon couples all controls through the dynamics and is typically posed as a single large nonlinear program. The proof shows you can solve **$T-1$ sequences of one-step problems** instead: at each $(t,\mathbf{x})$ minimize

$$
\mathbf{u}\mapsto c_t(\mathbf{x},\mathbf{u}) + J_{t+1}^\star(\mathbf{f}_t(\mathbf{x},\mathbf{u})).
$$

In finite state–action spaces this becomes pure table lookup and argmin. In continuous spaces you still solve local one-step minimizations, but you avoid a monolithic horizon-coupled NLP.
```

```{prf:remark} Graph interpretation (optional intuition)
Unroll time to form a DAG whose nodes are $(t,\mathbf{x})$ and whose edges correspond to feasible controls with edge weight $c_t(\mathbf{x},\mathbf{u})$. The terminal node cost is $c_\mathrm{T}(\cdot)$. The Bolza problem is a shortest-path problem on this DAG. The equation

$$
J_t^\star(\mathbf{x})=\min_{\mathbf{u}}\{c_t(\mathbf{x},\mathbf{u})+J_{t+1}^\star(\mathbf{f}_t(\mathbf{x},\mathbf{u}))\}
$$

is exactly the dynamic programming recursion for shortest paths on acyclic graphs, hence backward induction is optimal.
```

<!-- ```{prf:remark} If minimizers may not exist
Replace each "min" by "inf" in the definitions and state that for every $\varepsilon>0$ there exist $\varepsilon$-optimal selectors $\boldsymbol{\pi}_t^\varepsilon(\cdot)$ achieving cost within $\varepsilon$ of $J_t^\star(\cdot)$. The same cut-and-paste and induction go through.
``` -->



### Example: Optimal Harvest in Resource Management

Dynamic programming is often used in resource management and conservation biology to devise policies to be implemented by decision makers and stakeholders : for eg. in fishereries, or timber harvesting. Per {cite}`Conroy2013`, we consider a population of a particular species, whose abundance we denote by $x_t$, where $t$ represents discrete time steps. Our objective is to maximize the cumulative harvest over a finite time horizon, while also considering the long-term sustainability of the population. This optimization problem can be formulated as:

$$
\text{maximize} \quad \sum_{t=t_0}^{t_f} F(x_t \cdot h_t) + F_\mathrm{T}(x_{t_f})
$$

Here, $F(\cdot)$ represents the immediate reward function associated with harvesting, $h_t$ is the harvest rate at time $t$, and $F_\mathrm{T}(\cdot)$ denotes a terminal value function that could potentially assign value to the final population state. In this particular problem, we assign no terminal value to the final population state, setting $F_\mathrm{T}(x_{t_f}) = 0$ and allowing us to focus solely on the cumulative harvest over the time horizon.

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

```{code-cell} python
:tags: [hide-input]

#  label: dp-harvest-policy
#  caption: Dynamic programming harvest example: printed output shows the optimal policy table, resulting population trajectory, and per-period harvests for an initial population of 50 fish.

%config InlineBackend.figure_format = 'retina'
import numpy as np

# Parameters
r_max = 0.3
K = 125
T = 20  # Number of time steps
N_max = 100  # Maximum population size to consider
h_max = 0.5  # Maximum harvest rate
h_step = 0.1  # Step size for harvest rate

# Create state and decision spaces
N_space = np.arange(1, N_max + 1)
h_space = np.arange(0, h_max + h_step, h_step)

# Initialize value function and policy
V = np.zeros((T + 1, len(N_space)))
policy = np.zeros((T, len(N_space)))

# Terminal value function (F_T)
def terminal_value(N):
    return 0

# State return function (F)
def state_return(N, h):
    return N * h

# State dynamics function
def state_dynamics(N, h):
    return N + r_max * N * (1 - N / K) - N * h

# Backward iteration
for t in range(T - 1, -1, -1):
    for i, N in enumerate(N_space):
        max_value = float('-inf')
        best_h = 0

        for h in h_space:
            if h > 1:  # Ensure harvest rate doesn't exceed 100%
                continue

            next_N = state_dynamics(N, h)
            if next_N < 1:  # Ensure population doesn't go extinct
                continue

            next_N_index = np.searchsorted(N_space, next_N)
            if next_N_index == len(N_space):
                next_N_index -= 1

            value = state_return(N, h) + V[t + 1, next_N_index]

            if value > max_value:
                max_value = value
                best_h = h

        V[t, i] = max_value
        policy[t, i] = best_h

# Function to simulate the optimal policy with conversion to Python floats
def simulate_optimal_policy(initial_N, T):
    trajectory = [float(initial_N)]  # Ensure first value is a Python float
    harvests = []

    for t in range(T):
        N = trajectory[-1]
        N_index = np.searchsorted(N_space, N)
        if N_index == len(N_space):
            N_index -= 1

        h = policy[t, N_index]
        harvests.append(float(N * h))  # Ensure harvest is a Python float

        next_N = state_dynamics(N, h)
        trajectory.append(float(next_N))  # Ensure next population value is a Python float

    return trajectory, harvests

# Example usage
initial_N = 50
trajectory, harvests = simulate_optimal_policy(initial_N, T)

print("Optimal policy:")
print(policy)
print("\nPopulation trajectory:", trajectory)
print("Harvests:", harvests)
print("Total harvest:", sum(harvests))

```


## Handling Continuous Spaces with Interpolation

In many real-world problems, such as our resource management example, the state space is inherently continuous. Dynamic programming, however, is usually defined on discrete state spaces. To reconcile this, we approximate the value function on a finite grid of points and use interpolation to estimate its value elsewhere.

In our earlier example, we acted as if population sizes could only be whole numbers: 1 fish, 2 fish, 3 fish. But real measurements don't fit neatly. What do you do with a survey that reports 42.7 fish? Our reflex in the code example was to round to the nearest integer, effectively saying "let's just call it 43." This corresponds to **nearest-neighbor interpolation**, also known as discretization. It's the zeroth-order case: you assume the value between grid points is constant and equal to the closest one. In practice, this amounts to overlaying a grid on the continuous landscape and forcing yourself to stand at the intersections. In our demo code, this step was carried out with [`numpy.searchsorted`](https://numpy.org/doc/2.0/reference/generated/numpy.searchsorted.html).

While easy to implement, nearest-neighbor interpolation can introduce artifacts:

1. Decisions may change abruptly, even if the state only shifts slightly.
2. Precision is lost, especially in regimes where small variations matter.
3. The curse of dimensionality forces an impractically fine grid if many state variables are added.

To address these issues, we can use **higher-order interpolation**. Instead of taking the nearest neighbor, we estimate the value at off-grid points by leveraging multiple nearby values.


### Backward Recursion with Interpolation

Suppose we have computed $J_{k+1}^\star(\mathbf{x})$ only at grid points $\mathbf{x} \in \mathcal{X}_\text{grid}$. 
To evaluate Bellman's equation at an arbitrary $\mathbf{x}_{k+1}$, we interpolate. 
Formally, let $I_{k+1}(\mathbf{x})$ be the interpolation operator that extends the value function from $\mathcal{X}_\text{grid}$ to the continuous space. Then:

$$
J_k^\star(\mathbf{x}_k) 
= \min_{\mathbf{u}_k} 
\Big[ c_k(\mathbf{x}_k, \mathbf{u}_k) 
+ I_{k+1}\big(\mathbf{f}_k(\mathbf{x}_k, \mathbf{u}_k)\big) \Big].
$$

For instance, in one dimension, linear interpolation gives:

$$
I_{k+1}(x) = J_{k+1}^\star(x_l) + \frac{x - x_l}{x_u - x_l} \big(J_{k+1}^\star(x_u) - J_{k+1}^\star(x_l)\big),
$$

where $x_l$ and $x_u$ are the nearest grid points bracketing $x$. Linear interpolation is often sufficient, but higher-order methods (cubic splines, radial basis functions) can yield smoother and more accurate estimates. The choice of interpolation scheme and grid layout both affect accuracy and efficiency. A finer grid improves resolution but increases computational cost, motivating strategies like adaptive grid refinement or replacing interpolation altogether with parametric function approximation which we are going to see later in this book.

In higher-dimensional spaces, naive interpolation becomes prohibitively expensive due to the curse of dimensionality. Several approaches such as tensorized multilinear interpolation, radial basis functions, and machine learning models address this challenge by extending a common principle: they approximate the value function at unobserved points using information from a finite set of evaluations. However, as dimensionality continues to grow, even tensor methods face scalability limits, which is why flexible parametric models like neural networks have become essential tools for high-dimensional function approximation.

```{prf:algorithm} Backward Recursion with Interpolation
:label: backward-recursion-interp

**Input:** 
- Terminal cost $c_\mathrm{T}(\cdot)$  
- Stage costs $c_t(\cdot,\cdot)$  
- Dynamics $f_t(\cdot,\cdot)$  
- Time horizon $T$  
- State grid $\mathcal{X}_\text{grid}$  
- Action set $\mathcal{U}$  
- Interpolation method $\mathcal{I}(\cdot)$ (e.g., linear, cubic spline, RBF, neural network)

**Output:** Value functions $J_t^\star(\cdot)$ and policies $\pi_t^\star(\cdot)$ for all $t$

1. **Initialize terminal values:**  
   - Compute $J_T^\star(\mathbf{x}) = c_\mathrm{T}(\mathbf{x})$ for all $\mathbf{x} \in \mathcal{X}_\text{grid}$  
   - Fit interpolator: $I_T \leftarrow \mathcal{I}(\{\mathbf{x}, J_T^\star(\mathbf{x})\}_{\mathbf{x} \in \mathcal{X}_\text{grid}})$

2. **Backward recursion:**  
   For $t = T-1, T-2, \dots, 0$:  
   
   a. **Bellman update at grid points:**  
      For each $\mathbf{x} \in \mathcal{X}_\text{grid}$:  
      - For each $\mathbf{u} \in \mathcal{U}$:  
        - Compute next state: $\mathbf{x}_\text{next} = f_t(\mathbf{x}, \mathbf{u})$  
        - **Interpolate future cost:** $\hat{J}_{t+1}(\mathbf{x}_\text{next}) = I_{t+1}(\mathbf{x}_\text{next})$  
        - Compute total cost: $J_t(\mathbf{x}, \mathbf{u}) = c_t(\mathbf{x}, \mathbf{u}) + \hat{J}_{t+1}(\mathbf{x}_\text{next})$  
      - **Minimize over actions:** $J_t^\star(\mathbf{x}) = \min_{\mathbf{u} \in \mathcal{U}} J_t(\mathbf{x}, \mathbf{u})$  
      - Store optimal action: $\pi_t^\star(\mathbf{x}) = \arg\min_{\mathbf{u} \in \mathcal{U}} J_t(\mathbf{x}, \mathbf{u})$
   
   b. **Fit interpolator for current stage:**  
      $I_t \leftarrow \mathcal{I}(\{\mathbf{x}, J_t^\star(\mathbf{x})\}_{\mathbf{x} \in \mathcal{X}_\text{grid}})$

3. **Return:** Interpolated value functions $\{I_t\}_{t=0}^T$ and policies $\{\pi_t^\star\}_{t=0}^{T-1}$
```


#### Example: Optimal Harvest with Linear Interpolation

Here is a demonstration of the backward recursion procedure using linear interpolation. 

```{code-cell} python
:tags: [hide-input]

#  label: dp-harvest-interp
#  caption: Backward recursion with linear interpolation: console output summarizes the smoothed optimal policy, state trajectory, and harvest totals for the resource management example.


import numpy as np

# Parameters
r_max = 0.3
K = 125
T = 20  # Number of time steps
N_max = 100  # Maximum population size to consider
h_max = 0.5  # Maximum harvest rate
h_step = 0.1  # Step size for harvest rate

# Create state and decision spaces
N_space = np.arange(1, N_max + 1)
h_space = np.arange(0, h_max + h_step, h_step)

# Initialize value function and policy
V = np.zeros((T + 1, len(N_space)))
policy = np.zeros((T, len(N_space)))

# Terminal value function (F_T)
def terminal_value(N):
    return 0

# State return function (F)
def state_return(N, h):
    return N * h

# State dynamics function
def state_dynamics(N, h):
    return N + r_max * N * (1 - N / K) - N * h

# Function to linearly interpolate between grid points in N_space
def interpolate_value_function(V, N_space, next_N, t):
    if next_N <= N_space[0]:
        return V[t, 0]  # Below or at minimum population, return minimum value
    if next_N >= N_space[-1]:
        return V[t, -1]  # Above or at maximum population, return maximum value
    
    # Find indices to interpolate between
    lower_idx = np.searchsorted(N_space, next_N) - 1
    upper_idx = lower_idx + 1
    
    # Linear interpolation
    N_lower = N_space[lower_idx]
    N_upper = N_space[upper_idx]
    weight = (next_N - N_lower) / (N_upper - N_lower)
    return (1 - weight) * V[t, lower_idx] + weight * V[t, upper_idx]

# Backward iteration with interpolation
for t in range(T - 1, -1, -1):
    for i, N in enumerate(N_space):
        max_value = float('-inf')
        best_h = 0
        
        for h in h_space:
            if h > 1:  # Ensure harvest rate doesn't exceed 100%
                continue
            
            next_N = state_dynamics(N, h)
            if next_N < 1:  # Ensure population doesn't go extinct
                continue
            
            # Interpolate value for next_N
            value = state_return(N, h) + interpolate_value_function(V, N_space, next_N, t + 1)
            
            if value > max_value:
                max_value = value
                best_h = h
        
        V[t, i] = max_value
        policy[t, i] = best_h

# Function to simulate the optimal policy using interpolation
def simulate_optimal_policy(initial_N, T):
    trajectory = [initial_N]
    harvests = []

    for t in range(T):
        N = trajectory[-1]
        
        # Interpolate optimal harvest rate
        if N <= N_space[0]:
            h = policy[t, 0]
        elif N >= N_space[-1]:
            h = policy[t, -1]
        else:
            lower_idx = np.searchsorted(N_space, N) - 1
            upper_idx = lower_idx + 1
            weight = (N - N_space[lower_idx]) / (N_space[upper_idx] - N_space[lower_idx])
            h = (1 - weight) * policy[t, lower_idx] + weight * policy[t, upper_idx]
        
        harvests.append(float(N * h))  # Ensure harvest is a Python float
        next_N = state_dynamics(N, h)
        trajectory.append(float(next_N))  # Ensure next population value is a Python float

    return trajectory, harvests

# Example usage
initial_N = 50
trajectory, harvests = simulate_optimal_policy(initial_N, T)

print("Optimal policy:")
print(policy)
print("\nPopulation trajectory:", trajectory)
print("Harvests:", harvests)
print("Total harvest:", sum(harvests))
```


Due to pedagogical considerations, this example is using our own implementation of the linear interpolation procedure. However, a more general and practical approach would be to use a built-in interpolation procedure in NumPy. Because our state space has a single dimension, we can simply use [scipy.interpolate.interp1d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html) which offers various interpolation methods through its `kind` argument, including 'linear', 'nearest', 'zero', 'slinear', 'quadratic', and 'cubic'.

Here's a more general implementation which here uses cubic interpolation through the `scipy.interpolate.interp1d` function: 

```{code-cell} python
:tags: [hide-input]

#  label: dp-harvest-cubic
#  caption: Cubic interpolation further smooths the optimal harvest policy. This output prints the leading rows of the policy table along with the resulting trajectory and harvest statistics.


import numpy as np
from scipy.interpolate import interp1d

rng = np.random.default_rng(2026)

# Parameters
r_max = 0.3
K = 125
T = 20  # Number of time steps
N_max = 100  # Maximum population size to consider
h_max = 0.5  # Maximum harvest rate
h_step = 0.1  # Step size for harvest rate

# Create state and decision spaces
N_space = np.arange(1, N_max + 1)
h_space = np.arange(0, h_max + h_step, h_step)

# Initialize value function and policy
V = np.zeros((T + 1, len(N_space)))
policy = np.zeros((T, len(N_space)))

# Terminal value function (F_T)
def terminal_value(N):
    return 0

# State return function (F)
def state_return(N, h):
    return N * h

# State dynamics function
def state_dynamics(N, h):
    return N + r_max * N * (1 - N / K) - N * h

# Function to create interpolation function for a given time step
def create_interpolator(V_t, N_space):
    return interp1d(N_space, V_t, kind='cubic', bounds_error=False, fill_value=(V_t[0], V_t[-1]))

# Backward iteration with interpolation
for t in range(T - 1, -1, -1):
    interpolator = create_interpolator(V[t+1], N_space)
    
    for i, N in enumerate(N_space):
        max_value = float('-inf')
        best_h = 0

        for h in h_space:
            if h > 1:  # Ensure harvest rate doesn't exceed 100%
                continue

            next_N = state_dynamics(N, h)
            if next_N < 1:  # Ensure population doesn't go extinct
                continue

            # Use interpolation to get the value for next_N
            value = state_return(N, h) + interpolator(next_N)

            if value > max_value:
                max_value = value
                best_h = h

        V[t, i] = max_value
        policy[t, i] = best_h

# Function to simulate the optimal policy using interpolation
def simulate_optimal_policy(initial_N, T):
    trajectory = [initial_N]
    harvests = []

    for t in range(T):
        N = trajectory[-1]
        
        # Create interpolator for the policy at time t
        policy_interpolator = interp1d(N_space, policy[t], kind='cubic', bounds_error=False, fill_value=(policy[t][0], policy[t][-1]))
        
        h = policy_interpolator(N)
        harvests.append(float(N * h))  # Ensure harvest is a Python float

        next_N = state_dynamics(N, h)
        trajectory.append(float(next_N))  # Ensure next population value is a Python float

    return trajectory, harvests

# Example usage
initial_N = 50
trajectory, harvests = simulate_optimal_policy(initial_N, T)

print("Optimal policy (first few rows):")
print(policy[:5])
print("\nPopulation trajectory:", trajectory)
print("Harvests:", harvests)
print("Total harvest:", sum(harvests))
```



## Linear Quadratic Regulator via Dynamic Programming

Linear dynamics and quadratic costs give a backward recursion that can be solved in closed form. The value function remains quadratic at every stage, and the optimal policy is a linear feedback law. No state grid, interpolation scheme, or general function approximator is needed. The recursion tracks a finite sequence of matrices.

Consider a discrete-time linear system:

$$
\mathbf{x}_{t+1} = A_t\mathbf{x}_t + B_t\mathbf{u}_t
$$

where $\mathbf{x}_t \in \mathbb{R}^n$ is the state and $\mathbf{u}_t \in \mathbb{R}^m$ is the control input. The matrices $A_t \in \mathbb{R}^{n \times n}$ and $B_t \in \mathbb{R}^{n \times m}$ describe the system dynamics at time $t$.

The cost function to be minimized is quadratic:

$$
J = \frac{1}{2}\mathbf{x}_T^\top Q_T \mathbf{x}_T + \frac{1}{2}\sum_{t=0}^{T-1} \left(\mathbf{x}_t^\top Q_t \mathbf{x}_t + \mathbf{u}_t^\top R_t \mathbf{u}_t\right)
$$

where $Q_T \succeq 0$ (positive semidefinite), $Q_t \succeq 0$, and $R_t \succ 0$ (positive definite) are symmetric matrices of appropriate dimensions. The positive definiteness of $R_t$ ensures the minimization problem is well-posed.

A quadratic terminal cost implies a quadratic value function at every earlier stage. Suppose the optimal cost-to-go at stage $t$ has the form

$$
J_t^\star(\mathbf{x}_t) = \frac{1}{2}\mathbf{x}_t^\top P_t \mathbf{x}_t
$$

for some positive semidefinite matrix $P_t$. At the terminal time, this is true by definition: $P_T = Q_T$.

Backward induction verifies the hypothesis. Assume $J_{t+1}^\star(\mathbf{x}_{t+1}) = \frac{1}{2}\mathbf{x}_{t+1}^\top P_{t+1} \mathbf{x}_{t+1}$. Bellman's equation at stage $t$ is

$$
J_t^\star(\mathbf{x}_t) = \min_{\mathbf{u}_t} \left[ \frac{1}{2}\mathbf{x}_t^\top Q_t \mathbf{x}_t + \frac{1}{2}\mathbf{u}_t^\top R_t \mathbf{u}_t + J_{t+1}^\star(\mathbf{x}_{t+1}) \right]
$$

Substituting the dynamics $\mathbf{x}_{t+1} = A_t\mathbf{x}_t + B_t\mathbf{u}_t$ and the quadratic form for $J_{t+1}^\star$:

$$
J_t^\star(\mathbf{x}_t) = \min_{\mathbf{u}_t} \left[ \frac{1}{2}\mathbf{x}_t^\top Q_t \mathbf{x}_t + \frac{1}{2}\mathbf{u}_t^\top R_t \mathbf{u}_t + \frac{1}{2}(A_t\mathbf{x}_t + B_t\mathbf{u}_t)^\top P_{t+1} (A_t\mathbf{x}_t + B_t\mathbf{u}_t) \right]
$$

Expanding the last term:

$$
(A_t\mathbf{x}_t + B_t\mathbf{u}_t)^\top P_{t+1} (A_t\mathbf{x}_t + B_t\mathbf{u}_t) = \mathbf{x}_t^\top A_t^\top P_{t+1} A_t \mathbf{x}_t + 2\mathbf{x}_t^\top A_t^\top P_{t+1} B_t \mathbf{u}_t + \mathbf{u}_t^\top B_t^\top P_{t+1} B_t \mathbf{u}_t
$$

The expression inside the minimization becomes:

$$
\frac{1}{2}\mathbf{x}_t^\top Q_t \mathbf{x}_t + \frac{1}{2}\mathbf{u}_t^\top R_t \mathbf{u}_t + \frac{1}{2}\mathbf{x}_t^\top A_t^\top P_{t+1} A_t \mathbf{x}_t + \mathbf{x}_t^\top A_t^\top P_{t+1} B_t \mathbf{u}_t + \frac{1}{2}\mathbf{u}_t^\top B_t^\top P_{t+1} B_t \mathbf{u}_t
$$

Collecting terms involving $\mathbf{u}_t$:

$$
= \frac{1}{2}\mathbf{x}_t^\top (Q_t + A_t^\top P_{t+1} A_t) \mathbf{x}_t + \mathbf{x}_t^\top A_t^\top P_{t+1} B_t \mathbf{u}_t + \frac{1}{2}\mathbf{u}_t^\top (R_t + B_t^\top P_{t+1} B_t) \mathbf{u}_t
$$

This is a quadratic function of $\mathbf{u}_t$. To find the minimizer, we take the gradient with respect to $\mathbf{u}_t$ and set it to zero:

$$
\frac{\partial}{\partial \mathbf{u}_t} = (R_t + B_t^\top P_{t+1} B_t) \mathbf{u}_t + B_t^\top P_{t+1} A_t \mathbf{x}_t = 0
$$

Since $R_t + B_t^\top P_{t+1} B_t$ is positive definite (both $R_t$ and $P_{t+1}$ are positive semidefinite with $R_t$ strictly positive), we can solve for the optimal control:

$$
\mathbf{u}_t^\star = -(R_t + B_t^\top P_{t+1} B_t)^{-1} B_t^\top P_{t+1} A_t \mathbf{x}_t
$$

Define the gain matrix:

$$
K_t = (R_t + B_t^\top P_{t+1} B_t)^{-1} B_t^\top P_{t+1} A_t
$$

so that $\mathbf{u}_t^\star = -K_t\mathbf{x}_t$. This is a **linear feedback policy**: the optimal control is simply a linear function of the current state.

Substituting $\mathbf{u}_t^\star$ back into the cost-to-go expression and simplifying (by completing the square), we obtain:

$$
J_t^\star(\mathbf{x}_t) = \frac{1}{2}\mathbf{x}_t^\top P_t \mathbf{x}_t
$$

where $P_t$ satisfies the **discrete-time Riccati equation**:

$$
P_t = Q_t + A_t^\top P_{t+1} A_t - A_t^\top P_{t+1} B_t (R_t + B_t^\top P_{t+1} B_t)^{-1} B_t^\top P_{t+1} A_t
$$


The resulting backward recursion is:


````{prf:algorithm} Backward Recursion for LQR
:label: backward-recursion-lqr

**Input:** System matrices $A_t, B_t$, cost matrices $Q_t, R_t, Q_T$, time horizon $T$

**Output:** Cost matrices $P_t$ and gain matrices $K_t$ for $t = 0, \ldots, T-1$

1. **Initialize:** $P_T = Q_T$

2. **For** $t = T-1, T-2, \ldots, 0$:
   1. Compute the gain matrix:

      $$K_t = (R_t + B_t^\top P_{t+1} B_t)^{-1} B_t^\top P_{t+1} A_t$$

   2. Compute the cost matrix via the Riccati equation:

      $$P_t = Q_t + A_t^\top P_{t+1} A_t - A_t^\top P_{t+1} B_t (R_t + B_t^\top P_{t+1} B_t)^{-1} B_t^\top P_{t+1} A_t$$

3. **End For**

4. **Return:** $\{P_0, \ldots, P_T\}$ and $\{K_0, \ldots, K_{T-1}\}$

**Optimal policy:** $\mathbf{u}_t^\star = -K_t\mathbf{x}_t$

**Optimal cost-to-go:** $J_t^\star(\mathbf{x}_t) = \frac{1}{2}\mathbf{x}_t^\top P_t \mathbf{x}_t$
````

### Local Stabilization of the Cart-Pole

The cart-pole in the trajectory-optimization chapter started from the downward configuration and required a large nonlinear maneuver to reach the top. Once it is near the upright equilibrium, a smaller problem remains: reject local deviations by moving the cart in response to the measured state. The nonlinear state and input are the same as before,

$$
\mathbf{x}=(p,v,\theta,\omega), \qquad
u=\text{horizontal cart acceleration},
$$

with $\theta=0$ at upright. Linearizing the discrete RK4 update $F_h$ at $(\mathbf{x}^\star,u^\star)=(0,0)$ gives

$$
\delta\mathbf{x}_{k+1}
\approx A\,\delta\mathbf{x}_k+B\,\delta u_k,
\qquad
A=\left.\frac{\partial F_h}{\partial\mathbf{x}}\right|_{(0,0)},
\quad
B=\left.\frac{\partial F_h}{\partial u}\right|_{(0,0)}.
$$

The experiment uses $h=0.02$ s, $Q=\operatorname{diag}(2,0.2,80,3)$, and $R=0.15$. For this time-invariant infinite-horizon case, the Riccati recursion converges to a fixed matrix $P$ that satisfies the discrete algebraic Riccati equation. The corresponding policy is $u_k=-K\delta\mathbf{x}_k$.

The unconstrained linear closed loop is asymptotically stable when every eigenvalue of $A-BK$ lies inside the unit disk. The physical implementation adds two constraints that are absent from that eigenvalue calculation: acceleration is clipped to $|u|\leq 8\;\mathrm{m\,s^{-2}}$, and the cart must remain inside a $2.4$ m rail. Three deterministic nonlinear rollouts distinguish the claims supported by the linearization:

1. An uncontrolled pole begins $5^\circ$ from upright.
2. The LQR controller begins from the same $5^\circ$ displacement.
3. The same controller begins $45^\circ$ from upright with the same actuator and rail limits.

```{code-cell} python
:tags: [remove-cell]

from pathlib import Path
import sys

code_directory = Path.cwd() / "code"
if str(code_directory) not in sys.path:
    sys.path.insert(0, str(code_directory))

from cartpole_control import (
    CartPoleParameters,
    design_lqr,
    format_lqr_metrics,
    make_lqr_animation,
    make_lqr_figure,
    run_lqr_cases,
)

cartpole_parameters = CartPoleParameters()
lqr_design = design_lqr(cartpole_parameters)
lqr_cases = run_lqr_cases(cartpole_parameters)
```

```{code-cell} python
:label: fig-cartpole-lqr-limits
:caption: The discrete linear closed loop is stable, and the nonlinear controller recovers from a 5 degree perturbation. Without control, the same initial displacement grows. From 45 degrees, the commanded acceleration saturates and the cart reaches its rail limit, so the local controller does not complete the recovery. All curves use the same nonlinear plant; only the initial state and controller differ.
:tags: [remove-input]

print(format_lqr_metrics(lqr_cases, lqr_design))
make_lqr_figure(lqr_cases, cartpole_parameters)
```

The closed-loop eigenvalues certify asymptotic stability of the unconstrained linearized model, not every trajectory of the nonlinear constrained plant. The $5^\circ$ rollout remains in a region where the linear approximation supplies useful actions. The $45^\circ$ rollout immediately asks for more acceleration than the actuator can provide, then exhausts the available rail. Increasing the entries of $Q$ cannot remove those physical limits.

```{code-cell} python
:label: anim-cartpole-lqr
:caption: Nonlinear validation of the local LQR controller. The left panel shows the uncontrolled fall from 5 degrees, the center panel shows recovery from the same state, and the right panel shows the 45 degree rollout ending at the rail limit. Python generates each frame from the recorded trajectories.
:tags: [remove-input]

from IPython.display import HTML, display
import matplotlib.pyplot as plt

lqr_animation = make_lqr_animation(lqr_cases, cartpole_parameters)
display(HTML(lqr_animation.to_jshtml()))
plt.close(lqr_animation._fig)
```

Balancing a pen on a finger motivates the action channel because the finger stabilizes the object by moving its base. The cart-pole model replaces the contact by a planar frictionless hinge. A real pen can slip, detach, flex, and rotate out of the plane, while sensing and hand motion introduce delays. The calculation establishes local stabilization for the stated rigid-body model; the classroom demonstration shares its instability and feedback mechanism, not all of its equations.

:::{dropdown} Inspect the linearization and LQR design
```{literalinclude} code/cartpole_control.py
:language: python
:start-at: def linearize_upright
:end-before: def simulate_lqr
:linenos:
```
:::

{download}`Download the shared nonlinear cart-pole and LQR source <code/cartpole_control.py>`.

# Stochastic Dynamic Programming and Markov Decision Processes

While our previous discussion centered on deterministic systems, many real-world problems involve uncertainty. Stochastic Dynamic Programming (SDP) extends our framework to handle stochasticity in both the objective function and system dynamics. This extension naturally leads us to consider more general policy classes and to formalize when simpler policies suffice.

## Decision Rules and Policies

Before diving into stochastic systems, we need to establish terminology for the different types of strategies a decision maker might employ. In the deterministic setting, we implicitly used feedback controllers of the form $u(\mathbf{x}, t)$. In the stochastic setting, we must be more precise about what information policies can use and how they select actions.

A **decision rule** is a prescription for action selection in each state at a specified decision epoch. These rules can vary in their complexity based on two main criteria:

1. **Dependence on history**: Markovian or History-dependent
2. **Action selection method**: Deterministic or Randomized

**Markovian decision rules** depend only on the current state, while **history-dependent rules** consider the entire sequence of past states and actions. Formally, a history $h_t$ at time $t$ is:

$$h_t = (s_1, a_1, \ldots, s_{t-1}, a_{t-1}, s_t)$$

The set of all possible histories at time $t$, denoted $H_t$, grows exponentially with $t$:
- $H_1 = \mathcal{S}$ (just the initial state)
- $H_2 = \mathcal{S} \times \mathcal{A} \times \mathcal{S}$
- $H_t = \mathcal{S} \times (\mathcal{A} \times \mathcal{S})^{t-1}$

**Deterministic rules** select an action with certainty, while **randomized rules** specify a probability distribution over the action space.

These classifications lead to four types of decision rules:
1. **Markovian Deterministic (MD)**: $\pi_t: \mathcal{S} \rightarrow \mathcal{A}_s$
2. **Markovian Randomized (MR)**: $\pi_t: \mathcal{S} \rightarrow \mathcal{P}(\mathcal{A}_s)$
3. **History-dependent Deterministic (HD)**: $\pi_t: H_t \rightarrow \mathcal{A}_s$
4. **History-dependent Randomized (HR)**: $\pi_t: H_t \rightarrow \mathcal{P}(\mathcal{A}_s)$

where $\mathcal{P}(\mathcal{A}_s)$ denotes the set of probability distributions over $\mathcal{A}_s$.

A **policy** $\boldsymbol{\pi}$ is a sequence of decision rules, one for each decision epoch:

$$\boldsymbol{\pi} = (\pi_1, \pi_2, ..., \pi_{N-1})$$

The set of all policies of class $K$ (where $K \in \{HR, HD, MR, MD\}$) is denoted as $\Pi^K$. These policy classes form a hierarchy:

$$\Pi^{MD} \subset \Pi^{MR} \subset \Pi^{HR}, \quad \Pi^{MD} \subset \Pi^{HD} \subset \Pi^{HR}$$

The largest set $\Pi^{HR}$ contains all possible policies. We ask: under what conditions can we restrict attention to the much simpler set $\Pi^{MD}$ without loss of optimality?

```{admonition} Notation: rules vs. policies
:class: tip
- **Decision rule (kernel).** A map from information to action distributions:
  - Markov, deterministic:  $\pi_t:\mathcal{S}\to\mathcal{A}_s$
  - Markov, randomized:     $\pi_t(\cdot\mid s)\in\Delta(\mathcal{A}_s)$
  - History-dependent:       $\pi_t(\cdot\mid h_t)\in\Delta(\mathcal{A}_{s_t})$
- **Policy (sequence).** $\boldsymbol{\pi}=(\pi_1,\pi_2,\ldots)$.
- **Stationary policy.** $\boldsymbol{\pi}=\mathrm{const}(\pi)$ with $\pi_t\equiv\pi \ \forall t$.  
  By convention, we identify $\pi$ with its stationary policy $\mathrm{const}(\pi)$ when no confusion arises.
```

## Stochastic System Dynamics

In the stochastic setting, our system evolution takes the form:

$$ \mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t) $$

Here, $\mathbf{w}_t$ represents a random disturbance or noise term at time $t$ due to the inherent uncertainty in the system's behavior. The stage cost function may also incorporate stochastic influences:

$$ c_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t) $$

In this context, our objective shifts from minimizing a deterministic cost to minimizing the expected total cost:

$$ \mathbb{E}\left[c_\mathrm{T}(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t, \mathbf{u}_t, \mathbf{w}_t)\right] $$

where the expectation is taken over the distributions of the random variables $\mathbf{w}_t$. The principle of optimality still holds in the stochastic case, but Bellman's optimality equation now involves an expectation:

$$ J_k^\star(\mathbf{x}_k) = \min_{\mathbf{u}_k} \mathbb{E}_{\mathbf{w}_k}\left[c_k(\mathbf{x}_k, \mathbf{u}_k, \mathbf{w}_k) + J_{k+1}^\star(\mathbf{f}_k(\mathbf{x}_k, \mathbf{u}_k, \mathbf{w}_k))\right] $$

In practice, this expectation is often computed by discretizing the distribution of $\mathbf{w}_k$ when the set of possible disturbances is very large or even continuous. Let's say we approximate the distribution with $K$ discrete values $\mathbf{w}_k^i$, each occurring with probability $p_k^i$. Then our Bellman equation becomes:

$$ J_k^\star(\mathbf{x}_k) = \min_{\mathbf{u}_k} \sum_{i=1}^K p_k^i \left(c_k(\mathbf{x}_k, \mathbf{u}_k, \mathbf{w}_k^i) + J_{k+1}^\star(\mathbf{f}_k(\mathbf{x}_k, \mathbf{u}_k, \mathbf{w}_k^i))\right) $$

## Optimality Equations in the Stochastic Setting

When dealing with stochastic systems, a central question arises: what information should our control policy use? In the most general case, a policy might use the entire history of observations and actions. However, as we'll see, the Markovian structure of our problems allows for dramatic simplifications.

Let $h_t = (s_1, a_1, s_2, a_2, \ldots, s_{t-1}, a_{t-1}, s_t)$ denote the complete history up to time $t$. In the stochastic setting, the history-based optimality equations become:

$$
u_t(h_t) = \sup_{a\in A_{s_t}}\left\{ r_t(s_t,a) + \sum_{j\in S} p_t(j\mid s_t,a)\, u_{t+1}(h_t,a,j) \right\},\quad u_N(h_N)=r_N(s_N)
$$

where we now explicitly use the transition probabilities $p_t(j|s_t,a)$ rather than a deterministic dynamics function.

````{prf:theorem} Principle of optimality for stochastic systems
:label: stoch-principle-opt

Let $u_t^*$ be the optimal expected return from epoch $t$ onward. Then:

**a.** $u_t^*$ satisfies the optimality equations:

$$u_t^*(h_t) = \sup_{a \in A_{s_t}} \left\{ r_t(s_t, a) + \sum_{j \in S} p_t(j|s_t, a) u_{t+1}^*(h_t, a, j) \right\}$$

with boundary condition $u_N^*(h_N) = r_N(s_N)$.

**b.** Any policy $\pi^*$ that selects actions attaining the supremum (or maximum) in the above equation at each history is optimal.
````

**Intuition:** This formalizes Bellman's principle of optimality: "An optimal policy has the property that whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision." The recursive structure means that optimal local decisions (choosing the best action at each step) lead to global optimality, even with uncertainty captured by the transition probabilities.

A simplification occurs when we examine these history-based equations more closely. The Markov property of our system dynamics and rewards means that the optimal return actually depends on the history only through the current state:

````{prf:proposition} State sufficiency for stochastic MDPs
:label: stoch-state-suff

In finite-horizon stochastic MDPs with Markovian dynamics and rewards, the optimal return $u_t^*(h_t)$ depends on the history only through the current state $s_t$. Thus we can write $u_t^*(h_t) = v_t^*(s_t)$ for some function $v_t^*$ that depends only on state and time.
````

````{prf:proof}
Following {cite:t}`Puterman1994` Theorem 4.4.2. We proceed by backward induction.

**Base case:** At the terminal time $N$, we have $u_N^*(h_N) = r_N(s_N)$ by the boundary condition. Since the terminal reward depends only on the final state $s_N$ and not on how we arrived there, $u_N^*(h_N) = u_N^*(s_N)$.

**Inductive step:** Assume $u_{t+1}^*(h_{t+1})$ depends on $h_{t+1}$ only through $s_{t+1}$ for all $t+1, \ldots, N$. Then from the optimality equation:

$$u_t^*(h_t) = \sup_{a \in A_{s_t}} \left\{ r_t(s_t, a) + \sum_{j \in S} p_t(j|s_t, a) u_{t+1}^*(h_t, a, j) \right\}$$

By the induction hypothesis, $u_{t+1}^*(h_t, a, j)$ depends only on the next state $j$, so:

$$u_t^*(h_t) = \sup_{a \in A_{s_t}} \left\{ r_t(s_t, a) + \sum_{j \in S} p_t(j|s_t, a) u_{t+1}^*(j) \right\}$$

Since the expression in brackets depends on $h_t$ only through the current state $s_t$ (the rewards and transition probabilities are Markovian), we conclude that $u_t^*(h_t) = u_t^*(s_t)$.
````

**Intuition:** The Markov property means that the current state contains all information needed to predict future evolution. The past provides no additional value for decision-making. This result allows us to work with value functions $v_t^*(s)$ indexed only by state and time, dramatically simplifying both theory and computation.

This state-sufficiency result, combined with the fact that randomization never helps when maximizing expected returns, leads to a dramatic simplification of the policy space:

````{prf:theorem} Policy reduction for stochastic MDPs
:label: stoch-policy-reduction

For finite-horizon stochastic MDPs with finite state and action sets:

$$
\sup_{\pi \in \Pi^{\mathrm{HR}}} v_\pi(s,t) = \max_{\pi \in \Pi^{\mathrm{MD}}} v_\pi(s,t)
$$

That is, there exists an optimal policy that is both deterministic and Markovian.
````

````{prf:proof}
Sketch following {cite:t}`Puterman1994` Lemma 4.3.1 and Theorem 4.4.2. First, Lemma 4.3.1 shows that for any function $w$ and any distribution $q$ over actions, $\sup_a w(a) \ge \sum_a q(a) w(a)$. Thus randomization cannot improve the expected value over choosing a single maximizing action. Second, by state sufficiency (Proposition {ref}`stoch-state-suff` and {cite:t}`Puterman1994` Thm. 4.4.2(a)), the optimal return depends on the history only through $(s_t,t)$. Therefore, selecting at each $(s_t,t)$ an action that attains the maximum yields a deterministic Markov decision rule which is optimal whenever the maximum is attained. If only a supremum exists, $\varepsilon$-optimal selectors exist by choosing actions within $\varepsilon$ of the supremum (see {cite:t}`Puterman1994` Thm. 4.3.4).
````

**Intuition:** Even in stochastic systems, randomization in the policy doesn't help when maximizing expected returns: you should always choose the action with the highest expected value. Combined with state sufficiency, this means simple state-to-action mappings are optimal.

These results justify focusing on deterministic Markov policies and lead to the backward recursion algorithm for stochastic systems: 

````{prf:algorithm} Backward Recursion for Stochastic Dynamic Programming
:label: backward-recursion-stochastic

**Input:** Terminal cost function $c_\mathrm{T}(\cdot)$, stage cost functions $c_t(\cdot, \cdot, \cdot)$, system dynamics $\mathbf{f}_t(\cdot, \cdot, \cdot)$, time horizon $\mathrm{T}$, disturbance distributions

**Output:** Optimal value functions $J_t^\star(\cdot)$ and optimal control policies $\pi_t^\star(\cdot)$ for $t = 1, \ldots, T$

1. Initialize $J_T^\star(\mathbf{x}) = c_\mathrm{T}(\mathbf{x})$ for all $\mathbf{x}$ in the state space
2. For $t = T-1, T-2, \ldots, 1$:
   1. For each state $\mathbf{x}$ in the state space:
      1. Compute $J_t^\star(\mathbf{x}) = \min_{\mathbf{u}} \mathbb{E}_{\mathbf{w}_t}\left[c_t(\mathbf{x}, \mathbf{u}, \mathbf{w}_t) + J_{t+1}^\star(\mathbf{f}_t(\mathbf{x}, \mathbf{u}, \mathbf{w}_t))\right]$
      2. Compute $\pi_t^\star(\mathbf{x}) = \arg\min_{\mathbf{u}} \mathbb{E}_{\mathbf{w}_t}\left[c_t(\mathbf{x}, \mathbf{u}, \mathbf{w}_t) + J_{t+1}^\star(\mathbf{f}_t(\mathbf{x}, \mathbf{u}, \mathbf{w}_t))\right]$
   2. End For
3. End For
4. Return $J_t^\star(\cdot)$, $\pi_t^\star(\cdot)$ for $t = 1, \ldots, T$
````

While SDP provides us with a framework to for handling uncertainty, it makes the curse of dimensionality even more difficult to handle in practice. Both the state space and the disturbance space must be discretized. This can lead to a combinatorial explosion in the number of scenarios to be evaluated at each stage.

However, just as we tackled the challenges of continuous state spaces with discretization and interpolation, we can devise efficient methods to handle the additional complexity of evaluating expectations. This problem essentially becomes one of numerical integration. When the set of disturbances is continuous (as is often the case with continuous state spaces), we enter a domain where numerical quadrature methods could be applied. But these methods tend to scale poorly as the number of dimensions grows. This is where more efficient techniques, often rooted in Monte Carlo methods, come into play. Two ingredients tackle the curse of dimensionality:

1. Function approximation (through discretization, interpolation, neural networks, etc.)
2. Monte Carlo integration (simulation)

These two elements essentially distill the key ingredients of machine learning, which is the direction we'll be exploring in this course. 

## Example: Stochastic Optimal Harvest in Resource Management

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

By expressing the harvest rate as a random variable, we mean to capture the fact that harvesting is a not completely under our control: we might obtain more or less what we had intended to. Furthermore, we generalize the population dynamics to the stochastic case via: 

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

```{code-cell} python
:tags: [hide-input]

#  label: dp-harvest-stochastic
#  caption: Stochastic resource management simulation: the cell reports the optimal policy sample, average trajectory, and visualizes ensemble trajectories plus the distribution of total harvest.

import numpy as np
from scipy.interpolate import interp1d

# Parameters
r_max = 0.3
K = 125
T = 30  # Number of time steps
N_max = 100  # Maximum population size to consider
h_max = 0.5  # Maximum harvest rate
h_step = 0.1  # Step size for harvest rate

# Create state and decision spaces
N_space = np.linspace(1, N_max, 100)  # Using more granular state space
h_space = np.arange(0, h_max + h_step, h_step)

# Stochastic parameters
h_outcomes = np.array([0.75, 1.0, 1.25])
h_probs = np.array([0.25, 0.5, 0.25])
r_outcomes = np.array([0.85, 1.05, 1.15]) * r_max
r_probs = np.array([0.25, 0.5, 0.25])

# Initialize value function and policy
V = np.zeros((T + 1, len(N_space)))
policy = np.zeros((T, len(N_space)))

# State return function (F)
def state_return(N, h):
    return N * h

# State dynamics function (stochastic)
def state_dynamics(N, h, r):
    return N + r * N * (1 - N / K) - h * N

# Function to create interpolation function for a given time step
def create_interpolator(V_t, N_space):
    return interp1d(N_space, V_t, kind='linear', bounds_error=False, fill_value=(V_t[0], V_t[-1]))

# Backward iteration with stochastic dynamics
for t in range(T - 1, -1, -1):
    interpolator = create_interpolator(V[t+1], N_space)
    
    for i, N in enumerate(N_space):
        max_value = float('-inf')
        best_h = 0

        for h in h_space:
            if h > 1:  # Ensure harvest rate doesn't exceed 100%
                continue

            expected_value = 0
            for h_factor, h_prob in zip(h_outcomes, h_probs):
                for r_factor, r_prob in zip(r_outcomes, r_probs):
                    realized_h = h * h_factor
                    realized_r = r_factor

                    next_N = state_dynamics(N, realized_h, realized_r)
                    if next_N < 1:  # Ensure population doesn't go extinct
                        continue

                    # Use interpolation to get the value for next_N
                    value = state_return(N, realized_h) + interpolator(next_N)
                    expected_value += value * h_prob * r_prob

            if expected_value > max_value:
                max_value = expected_value
                best_h = h

        V[t, i] = max_value
        policy[t, i] = best_h

# Function to simulate the optimal policy using interpolation (stochastic version)
def simulate_optimal_policy(initial_N, T, num_simulations=100):
    all_trajectories = []
    all_harvests = []

    for _ in range(num_simulations):
        trajectory = [initial_N]
        harvests = []

        for t in range(T):
            N = trajectory[-1]
            
            # Create interpolator for the policy at time t
            policy_interpolator = interp1d(N_space, policy[t], kind='linear', bounds_error=False, fill_value=(policy[t][0], policy[t][-1]))
            
            intended_h = policy_interpolator(N)
            
            # Apply stochasticity
            h_factor = rng.choice(h_outcomes, p=h_probs)
            r_factor = rng.choice(r_outcomes, p=r_probs)
            
            realized_h = intended_h * h_factor
            harvests.append(N * realized_h)

            next_N = state_dynamics(N, realized_h, r_factor)
            trajectory.append(next_N)

        all_trajectories.append(trajectory)
        all_harvests.append(harvests)

    return all_trajectories, all_harvests

# Example usage
initial_N = 50
trajectories, harvests = simulate_optimal_policy(initial_N, T)

# Calculate average trajectory and total harvest
avg_trajectory = np.mean(trajectories, axis=0)
avg_total_harvest = np.mean([sum(h) for h in harvests])

print("Optimal policy (first few rows):")
print(policy[:5])
print("\nAverage population trajectory:", avg_trajectory)
print("Average total harvest:", avg_total_harvest)

# Plot results
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt

# Apply book style
try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
except (ImportError, OSError):
    pass  # Use matplotlib defaults

plt.figure(figsize=(12, 6))
plt.subplot(121)
for traj in trajectories[:20]:  # Plot first 20 trajectories
    plt.plot(range(T+1), traj, alpha=0.3)
plt.plot(range(T+1), avg_trajectory, 'r-', linewidth=2)
plt.title('Population Trajectories')
plt.xlabel('Time')
plt.ylabel('Population')

plt.subplot(122)
plt.hist([sum(h) for h in harvests], bins=20)
plt.title('Distribution of Total Harvest')
plt.xlabel('Total Harvest')
plt.ylabel('Frequency')

plt.tight_layout()
```



# Markov Decision Process Formulation

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

## Notation in Operations Reseach 

The presentation above was intended to bridge the gap between the control-theoretic perspective and the world of closed-loop control through the idea of determining the value function of a parametric optimal control problem. We then saw how the backward induction procedure was applicable to both the deterministic and stochastic cases by taking the expectation over the disturbance variable. We then said that we can alternatively work with a representation of our system where instead of writing our model as a deterministic dynamics function taking a disturbance as an input, we would rather work directly via its transition probability function, which gives rise to the Markov chain interpretation of our system in simulation.

Note that the notation used in control theory tends to differ from that found in operations research communities, in which the field of dynamic programming flourished. We summarize those (purely notational) differences in this section.

In operations research, the system state at each decision epoch is typically denoted by $s \in \mathcal{S}$, where $S$ is the set of possible system states. When the system is in state $s$, the decision maker may choose an action $a$ from the set of allowable actions $\mathcal{A}_s$. The union of all action sets is denoted as $\mathcal{A} = \bigcup_{s \in \mathcal{S}} \mathcal{A}_s$.

The dynamics of the system are described by a transition probability function $p_t(j | s, a)$, which represents the probability of transitioning to state $j \in \mathcal{S}$ at time $t+1$, given that the system is in state $s$ at time $t$ and action $a \in \mathcal{A}_s$ is chosen. This transition probability function satisfies:

$$\sum_{j \in \mathcal{S}} p_t(j | s, a) = 1$$

It's worth noting that in operations research, we typically work with reward maximization rather than cost minimization, which is more common in control theory. However, we can easily switch between these perspectives by simply negating the quantity. That is, maximizing a reward function is equivalent to minimizing its negative, which we would then call a cost function.

The reward function is denoted by $r_t(s, a)$, representing the reward received at time $t$ when the system is in state $s$ and action $a$ is taken. In some cases, the reward may also depend on the next state, in which case it is denoted as $r_t(s, a, j)$. The expected reward can then be computed as:

$$r_t(s, a) = \sum_{j \in \mathcal{S}} r_t(s, a, j) p_t(j | s, a)$$

Combined together, these elemetns specify a Markov decision process, which is fully described by the tuple:

$$\{T, S, \mathcal{A}_s, p_t(\cdot | s, a), r_t(s, a)\}$$

where $\mathrm{T}$ represents the set of decision epochs (the horizon). 

## What is an Optimal Policy?

Let's go back to the starting point and define what it means for a policy to be optimal in a Markov Decision Problem. For this, we will be considering different possible search spaces (policy classes) and compare policies based on the ordering of their value from any possible start state. The value of a policy $\boldsymbol{\pi}$ (optimal or not) is defined as the expected total reward obtained by following that policy from a given starting state. Formally, for a finite-horizon MDP with $N$ decision epochs, we define the value function $v^{\boldsymbol{\pi}}(s, t)$ as:

$$
v^{\boldsymbol{\pi}}(s, t) \triangleq \mathbb{E}\left[\sum_{k=t}^{N-1} r_t(S_k, A_k) + r_N(S_N) \mid S_t = s\right]
$$

where $S_t$ is the state at time $t$, $A_t$ is the action taken at time $t$, and $r_t$ is the reward function. For simplicity, we write $v^{\boldsymbol{\pi}}(s)$ to denote $v^{\boldsymbol{\pi}}(s, 1)$, the value of following policy $\boldsymbol{\pi}$ from state $s$ at the first stage over the entire horizon $N$.

In finite-horizon MDPs, our goal is to identify an optimal policy, denoted by $\boldsymbol{\pi}^*$, that maximizes total expected reward over the horizon $N$. Specifically:

$$
v^{\boldsymbol{\pi}^*}(s) \geq v^{\boldsymbol{\pi}}(s), \quad \forall s \in \mathcal{S}, \quad \forall \boldsymbol{\pi} \in \Pi^{\text{HR}}
$$

We call $\boldsymbol{\pi}^*$ an **optimal policy** because it yields the highest possible value across all states and all policies within the policy class $\Pi^{\text{HR}}$. We denote by $v^*$ the maximum value achievable by any policy:

$$
v^*(s) = \max_{\boldsymbol{\pi} \in \Pi^{\text{HR}}} v^{\boldsymbol{\pi}}(s), \quad \forall s \in \mathcal{S}
$$

In reinforcement learning literature, $v^*$ is typically referred to as the "optimal value function," while in some operations research references, it might be called the "value of an MDP." An optimal policy $\boldsymbol{\pi}^*$ is one for which its value function equals the optimal value function:

$$
v^{\boldsymbol{\pi}^*}(s) = v^*(s), \quad \forall s \in \mathcal{S}
$$

This notion of optimality applies to every state. Policies optimal in this sense are sometimes called "uniformly optimal policies." A weaker notion of optimality, often encountered in reinforcement learning practice, is optimality with respect to an initial distribution of states. In this case, we seek a policy $\boldsymbol{\pi} \in \Pi^{\text{HR}}$ that maximizes:

$$
\sum_{s \in \mathcal{S}} v^{\boldsymbol{\pi}}(s) P_1(S_1 = s)
$$

where $P_1(S_1 = s)$ is the probability of starting in state $s$.

The maximum value can be achieved by searching over the space of deterministic Markovian Policies. Consequently:

$$ v^*(s) = \max_{\boldsymbol{\pi} \in \Pi^{\mathrm{HR}}} v^{\boldsymbol{\pi}}(s) = \max _{\boldsymbol{\pi} \in \Pi^{M D}} v^{\boldsymbol{\pi}}(s), \quad s \in S$$

This equality significantly simplifies the computational complexity of our algorithms, as the search problem can now be decomposed into $N$ sub-problems in which we only have to search over the set of possible actions. This is the backward induction algorithm, which we present a second time, but departing this time from the control-theoretic notation and using the MDP formalism:  

````{prf:algorithm} Backward Induction
:label: backward-induction

**Input:** State space $S$, Action space $A$, Transition probabilities $p_t$, Reward function $r_t$, Time horizon $N$

**Output:** Optimal value functions $v^*$

1. Initialize:
   - Set $t = N$
   - For all $s_N \in S$:

     $$v^*(s_N, N) = r_N(s_N)$$

2. For $t = N-1$ to $1$:
   - For each $s_t \in S$:
     a. Compute the optimal value function:

        $$v^*(s_t, t) = \max_{a \in A_{s_t}} \left\{r_t(s_t, a) + \sum_{j \in S} p_t(j | s_t, a) v^*(j, t+1)\right\}$$
     
     b. Determine the set of optimal actions:

        $$A_{s_t,t}^* = \arg\max_{a \in A_{s_t}} \left\{r_t(s_t, a) + \sum_{j \in S} p_t(j | s_t, a) v^*(j, t+1)\right\}$$

3. Return the optimal value functions $u_t^*$ and optimal action sets $A_{s_t,t}^*$ for all $t$ and $s_t$
````

Note that the same procedure can also be used for finding the value of a policy with minor changes; 

````{prf:algorithm} Policy Evaluation
:label: backward-policy-evaluation

**Input:** 
- State space $S$
- Action space $A$
- Transition probabilities $p_t$
- Reward function $r_t$
- Time horizon $N$
- A markovian deterministic policy $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_{N-1})$

**Output:** Value function $v^{\boldsymbol{\pi}}$ for policy $\boldsymbol{\pi}$

1. Initialize:
   - Set $t = N$
   - For all $s_N \in S$:

     $$v^{\boldsymbol{\pi}}(s_N, N) = r_N(s_N)$$

2. For $t = N-1$ to $1$:
   - For each $s_t \in S$:
     a. Compute the value function for the given policy:

        $$v^{\boldsymbol{\pi}}(s_t, t) = r_t(s_t, \pi_t(s_t)) + \sum_{j \in S} p_t(j | s_t, \pi_t(s_t)) v^{\boldsymbol{\pi}}(j, t+1)$$

3. Return the value function $v^{\boldsymbol{\pi}}(s_t, t)$ for all $t$ and $s_t$
````

This code could also finally be adapted to support randomized policies using:

$$v^{\boldsymbol{\pi}}(s_t, t) = \sum_{a_t \in \mathcal{A}_{s_t}} \pi_t(a_t \mid s_t) \left( r_t(s_t, a_t) + \sum_{j \in S} p_t(j | s_t, a_t) v^{\boldsymbol{\pi}}(j, t+1) \right)$$


# Infinite-Horizon MDPs

It often makes sense to model control problems over infinite horizons. We extend the previous setting and define the expected total reward of policy $\boldsymbol{\pi} \in \Pi^{\mathrm{HR}}$, $v^{\boldsymbol{\pi}}$ as:

$$
v^{\boldsymbol{\pi}}(s) = \mathbb{E}\left[\sum_{t=1}^{\infty} r(S_t, A_t)\right]
$$

One drawback of this model is that we could easily encounter values that are $+\infty$ or $-\infty$, even in a setting as simple as a single-state MDP which loops back into itself and where the accrued reward is nonzero.

Therefore, it is often more convenient to work with an alternative formulation which guarantees the existence of a limit: the expected total discounted reward of policy $\boldsymbol{\pi} \in \Pi^{\mathrm{HR}}$ is defined to be:

$$
v_\gamma^{\boldsymbol{\pi}}(s) \equiv \lim_{N \rightarrow \infty} \mathbb{E}\left[\sum_{t=1}^N \gamma^{t-1} r(S_t, A_t)\right]
$$

for $0 \leq \gamma < 1$ and when $\max_{s \in \mathcal{S}} \max_{a \in \mathcal{A}_s}|r(s, a)| = R_{\max} < \infty$, in which case, $|v_\gamma^{\boldsymbol{\pi}}(s)| \leq (1-\gamma)^{-1} R_{\max}$.


Finally, another possibility for the infinite-horizon setting is the so-called average reward or gain of policy $\boldsymbol{\pi} \in \Pi^{\mathrm{HR}}$ defined as:

$$
g^{\boldsymbol{\pi}}(s) \equiv \lim_{N \rightarrow \infty} \frac{1}{N} \mathbb{E}\left[\sum_{t=1}^N r(S_t, A_t)\right]
$$

We won't be working with this formulation in this course due to its inherent practical and theoretical complexities. 

Extending the previous notion of optimality from finite-horizon models, a policy $\boldsymbol{\pi}^*$ is said to be discount optimal for a given $\gamma$ if: 

$$
v_\gamma^{\boldsymbol{\pi}^*}(s) \geq v_\gamma^{\boldsymbol{\pi}}(s) \quad \text { for each } s \in S \text { and all } \boldsymbol{\pi} \in \Pi^{\mathrm{HR}}
$$

Furthermore, the value of a discounted MDP $v_\gamma^*(s)$, is defined by:

$$
v_\gamma^*(s) \equiv \max _{\boldsymbol{\pi} \in \Pi^{\mathrm{HR}}} v_\gamma^{\boldsymbol{\pi}}(s)
$$

More often, we refer to $v_\gamma$ by simply calling it the optimal value function. 

As for the finite-horizon setting, the infinite horizon discounted model does not require history-dependent policies, since for any $\boldsymbol{\pi} \in \Pi^{HR}$ there exists a $\boldsymbol{\pi}^{\prime} \in \Pi^{MR}$ with identical total discounted reward:
$$
v_\gamma^*(s) \equiv \max_{\boldsymbol{\pi} \in \Pi^{HR}} v_\gamma^{\boldsymbol{\pi}}(s)=\max_{\boldsymbol{\pi} \in \Pi^{MR}} v_\gamma^{\boldsymbol{\pi}}(s) .
$$

## Random Horizon Interpretation of Discounting
The use of discounting can be motivated both from a modeling perspective and as a means to ensure that the total reward remains bounded. From the modeling perspective, we can view discounting as a way to weight more or less importance on the immediate rewards vs. the long-term consequences. There is also another interpretation which stems from that of a finite horizon model but with an uncertain end time. More precisely:

Let $v_\nu^{\boldsymbol{\pi}}(s)$ denote the expected total reward obtained by using policy $\boldsymbol{\pi}$ when the horizon length $\nu$ is random. We define it by:

$$
v_\nu^{\boldsymbol{\pi}}(s) \equiv \mathbb{E}_s^{\boldsymbol{\pi}}\left[\mathbb{E}_\nu\left\{\sum_{t=1}^\nu r(S_t, A_t)\right\}\right]
$$


````{prf:theorem} Random horizon interpretation of discounting
:label: prop-5-3-1
Suppose that the horizon $\nu$ follows a geometric distribution with parameter $\gamma$, $0 \leq \gamma < 1$, independent of the policy such that 
$P(\nu=n) = (1-\gamma) \gamma^{n-1}, \, n=1,2, \ldots$, then $v_\nu^{\boldsymbol{\pi}}(s) = v_\gamma^{\boldsymbol{\pi}}(s)$ for all $s \in \mathcal{S}$ .
````

````{prf:proof}
See proposition 5.3.1 in {cite}`Puterman1994`.

By definition of the finite-horizon value function and the law of total expectation:

$$
v_\nu^{\boldsymbol{\pi}}(s) = \sum_{n=1}^{\infty} P(\nu=n) \cdot v_n^{\boldsymbol{\pi}}(s) = \sum_{n=1}^{\infty} (1-\gamma) \gamma^{n-1} \cdot E_s^{\boldsymbol{\pi}} \left\{\sum_{t=1}^n r(S_t, A_t)\right\}.
$$

Combining the expectation with the sum over $n$:

$$
v_\nu^{\boldsymbol{\pi}}(s) = E_s^{\boldsymbol{\pi}} \left\{\sum_{n=1}^{\infty} (1-\gamma) \gamma^{n-1} \sum_{t=1}^n r(S_t, A_t)\right\}.
$$

**Reordering the summations:** Under the bounded reward assumption $|r(s,a)| \leq R_{\max}$ and $\gamma < 1$, we have

$$
E_s^{\boldsymbol{\pi}} \left\{\sum_{n=1}^{\infty} \sum_{t=1}^n |r(S_t, A_t)| \cdot (1-\gamma) \gamma^{n-1}\right\} \leq R_{\max} \sum_{n=1}^{\infty} n (1-\gamma) \gamma^{n-1} = \frac{R_{\max}}{1-\gamma} < \infty,
$$
which justifies exchanging the order of summation by Fubini's theorem.

To reverse the order, note that the pair $(n,t)$ with $1 \leq t \leq n$ can be reindexed by fixing $t$ first and letting $n$ range from $t$ to $\infty$:

$$
\sum_{n=1}^{\infty} \sum_{t=1}^n = \sum_{t=1}^{\infty} \sum_{n=t}^{\infty}.
$$

Therefore:
\begin{align*}
v_\nu^{\boldsymbol{\pi}}(s) &= E_s^{\boldsymbol{\pi}} \left\{\sum_{t=1}^{\infty} r(S_t, A_t) \sum_{n=t}^{\infty} (1-\gamma) \gamma^{n-1}\right\}.
\end{align*}

**Evaluating the inner sum:** Using the substitution $m = n - t + 1$ (so $n = m + t - 1$):
\begin{align*}
\sum_{n=t}^{\infty} (1-\gamma) \gamma^{n-1} &= \sum_{m=1}^{\infty} (1-\gamma) \gamma^{m+t-2} \\
&= \gamma^{t-1} (1-\gamma) \sum_{m=1}^{\infty} \gamma^{m-1} \\
&= \gamma^{t-1} (1-\gamma) \cdot \frac{1}{1-\gamma} = \gamma^{t-1}.
\end{align*}

Substituting back:

$$
v_\nu^{\boldsymbol{\pi}}(s) = E_s^{\boldsymbol{\pi}} \left\{\sum_{t=1}^{\infty} \gamma^{t-1} r(S_t, A_t)\right\} = v_\gamma^{\boldsymbol{\pi}}(s).
$$

````


## Vector Representation in Markov Decision Processes

Let V be the set of bounded real-valued functions on a discrete state space S. This means any function $ f \in V $ satisfies the condition:

$$
\|f\| = \max_{s \in S} |f(s)| < \infty.
$$
where notation $ \|f\| $ represents the sup-norm (or $ \ell_\infty $-norm) of the function $ f $. 

When working with discrete state spaces, we can interpret elements of V as vectors and linear operators on V as matrices, allowing us to leverage tools from linear algebra. The sup-norm ($\ell_\infty$ norm) of matrix $\mathbf{H}$ is defined as:

$$
\|\mathbf{H}\| \equiv \max_{s \in S} \sum_{j \in S} |\mathbf{H}_{s,j}|
$$

where $\mathbf{H}_{s,j}$ represents the $(s, j)$-th component of the matrix $\mathbf{H}$.

For a Markovian decision rule $\pi \in \Pi^{MD}$, we define:

\begin{align*}
\mathbf{r}_\pi(s) &\equiv r(s, \pi(s)), \quad \mathbf{r}_\pi \in \mathbb{R}^{|S|}, \\
[\mathbf{P}_\pi]_{s,j} &\equiv p(j \mid s, \pi(s)), \quad \mathbf{P}_\pi \in \mathbb{R}^{|S| \times |S|}.
\end{align*}

For a randomized decision rule $\pi \in \Pi^{MR}$, these definitions extend to:

\begin{align*}
\mathbf{r}_\pi(s) &\equiv \sum_{a \in A_s} \pi(a \mid s) \, r(s, a), \\
[\mathbf{P}_\pi]_{s,j} &\equiv \sum_{a \in A_s} \pi(a \mid s) \, p(j \mid s, a).
\end{align*}

In both cases, $\mathbf{r}_\pi$ denotes a reward vector in $\mathbb{R}^{|S|}$, with each component $\mathbf{r}_\pi(s)$ representing the reward associated with state $s$. Similarly, $\mathbf{P}_\pi$ is a transition probability matrix in $\mathbb{R}^{|S| \times |S|}$, capturing the transition probabilities under decision rule $\pi$.

For a nonstationary Markovian policy $\boldsymbol{\pi} = (\pi_1, \pi_2, \ldots) \in \Pi^{MR}$, the expected total discounted reward is given by:

$$
\mathbf{v}_\gamma^{\boldsymbol{\pi}}(s)=\mathbb{E}\left[\sum_{t=1}^{\infty} \gamma^{t-1} r\left(S_t, A_t\right) \,\middle|\, S_1 = s\right].
$$

Using vector notation, this can be expressed as:

$$
\begin{aligned}
\mathbf{v}_\gamma^{\boldsymbol{\pi}} &= \sum_{t=1}^{\infty} \gamma^{t-1} \mathbf{P}_{\boldsymbol{\pi}}^{t-1} \mathbf{r}_{\pi_1} \\
&= \mathbf{r}_{\pi_1} + \gamma \mathbf{P}_{\pi_1} \mathbf{r}_{\pi_2} + \gamma^2 \mathbf{P}_{\pi_1} \mathbf{P}_{\pi_2} \mathbf{r}_{\pi_3} + \cdots \\
&= \mathbf{r}_{\pi_1} + \gamma \mathbf{P}_{\pi_1} \left( \mathbf{r}_{\pi_2} + \gamma \mathbf{P}_{\pi_2} \mathbf{r}_{\pi_3} + \gamma^2 \mathbf{P}_{\pi_2} \mathbf{P}_{\pi_3} \mathbf{r}_{\pi_4} + \cdots \right).
\end{aligned}
$$

This formulation leads to a recursive relationship:

$$
\begin{align*}
\mathbf{v}_\gamma^{\boldsymbol{\pi}} &= \mathbf{r}_{\pi_1} + \gamma \mathbf{P}_{\pi_1} \mathbf{v}_\gamma^{\boldsymbol{\pi}^{\prime}}\\
&=\sum_{t=1}^{\infty} \gamma^{t-1} \mathbf{P}_{\boldsymbol{\pi}}^{t-1} \mathbf{r}_{\pi_t}
\end{align*}
$$

where $\boldsymbol{\pi}^{\prime} = (\pi_2, \pi_3, \ldots)$.


For a stationary policy $\boldsymbol{\pi} = \mathrm{const}(\pi)$ with constant decision rule $\pi$, the total expected reward simplifies to:

$$
\begin{align*}
\mathbf{v}_\gamma^{\pi} &= \mathbf{r}_\pi+ \gamma \mathbf{P}_\pi \mathbf{v}_\gamma^{\pi} \\
&=\sum_{t=1}^{\infty} \gamma^{t-1} \mathbf{P}_\pi^{t-1} \mathbf{r}_{\pi}
\end{align*}
$$

This last expression is called a Neumann series expansion, and it's guaranteed to exists under the assumptions of bounded reward and discount factor strictly less than one. 

```{prf:theorem} Neumann Series and Invertibility
:label: neumann-series

The **spectral radius** of a matrix $\mathbf{H}$ is defined as:

$$
\rho(\mathbf{H}) \equiv \max_{i} |\lambda_i(\mathbf{H})|
$$

where $\lambda_i(\mathbf{H})$ are the eigenvalues of $\mathbf{H}$.

**Neumann Series Existence:** For any matrix $\mathbf{H}$, the Neumann series

$$
\sum_{t=0}^{\infty} \mathbf{H}^t = \mathbf{I} + \mathbf{H} + \mathbf{H}^2 + \cdots
$$

converges if and only if $\rho(\mathbf{H}) < 1$. When this condition holds, the matrix $(\mathbf{I} - \mathbf{H})$ is invertible and

$$
(\mathbf{I} - \mathbf{H})^{-1} = \sum_{t=0}^{\infty} \mathbf{H}^t.
$$

```
Note that for any induced matrix norm $\|\cdot\|$ (i.e., a norm satisfying $\|\mathbf{H}\mathbf{v}\| \leq \|\mathbf{H}\| \cdot \|\mathbf{v}\|$ for all vectors $\mathbf{v}$) and any matrix $\mathbf{H}$, the spectral radius is bounded by:

$$
\rho(\mathbf{H}) \leq \|\mathbf{H}\|.
$$


This inequality provides a practical way to verify the convergence condition $\rho(\mathbf{H}) < 1$ by checking the simpler condition $\|\mathbf{H}\| < 1$ rather than trying to compute the eigenvalues directly.

We can now verify that $(\mathbf{I} - \gamma \mathbf{P}_\pi)$ is invertible and the Neumann series converges.

1. **Norm of the transition matrix:** Since $\mathbf{P}_\pi$ is a stochastic matrix (each row sums to 1 and all entries are non-negative), its $\ell_\infty$-norm is:

   $$
   \|\mathbf{P}_\pi\| = \max_{s \in S} \sum_{j \in S} [\mathbf{P}_\pi]_{s,j} = \max_{s \in S} 1 = 1.
   $$

2. **Norm of the scaled matrix:** Using the homogeneity property of norms, we have:

   $$
   \|\gamma \mathbf{P}_\pi\| = |\gamma| \cdot \|\mathbf{P}_\pi\| = |\gamma| \cdot 1 = |\gamma|.
   $$

3. **Bounding the spectral radius:** Since the spectral radius is bounded by the matrix norm:

   $$
   \rho(\gamma \mathbf{P}_\pi) \leq \|\gamma \mathbf{P}_\pi\| = |\gamma|.
   $$

4. **Verifying convergence:** Since $0 \leq \gamma < 1$ by assumption, we have:

   $$
   \rho(\gamma \mathbf{P}_\pi) \leq |\gamma| < 1.
   $$
   
   This strict inequality guarantees that $(\mathbf{I} - \gamma \mathbf{P}_\pi)$ is invertible and the Neumann series converges.

Therefore, the Neumann series expansion converges and yields:

$$
\mathbf{v}_\gamma^{\pi} = (\mathbf{I} - \gamma \mathbf{P}_\pi)^{-1} \mathbf{r}_\pi = \sum_{t=0}^{\infty} (\gamma \mathbf{P}_\pi)^t \mathbf{r}_\pi = \sum_{t=1}^{\infty} \gamma^{t-1} \mathbf{P}_\pi^{t-1} \mathbf{r}_\pi.
$$

Consequently, for a stationary policy, $\mathbf{v}_\gamma^{\pi}$ can be determined as the solution to the linear equation:

$$
\mathbf{v} = \mathbf{r}_\pi+ \gamma \mathbf{P}_\pi\mathbf{v},
$$

which can be rearranged to:

$$
(\mathbf{I} - \gamma \mathbf{P}_\pi) \mathbf{v} = \mathbf{r}_\pi.
$$

We can also characterize $\mathbf{v}_\gamma^{\pi}$ as the solution to an operator equation. More specifically, define the transformation $\mathrm{L}_\pi$ by

$$
\mathrm{L}_\pi \mathbf{v} \equiv \mathbf{r}_\pi+\gamma \mathbf{P}_\pi\mathbf{v}
$$

for any $\mathbf{v} \in V$. Intuitively, $\mathrm{L}_\pi$ takes a value function $\mathbf{v}$ as input and returns a new value function that combines immediate rewards ($\mathbf{r}_\pi$) with discounted future values ($\gamma \mathbf{P}_\pi\mathbf{v}$). 

```{note}
While we often refer to $\mathrm{L}_\pi$ as a "linear operator" in the RL literature, it is technically an **affine operator** (or affine transformation), not a linear operator in the strict sense. To see why, recall that a linear operator $\mathcal{T}$ must satisfy:

1. **Additivity:** $\mathcal{T}(\mathbf{v}_1 + \mathbf{v}_2) = \mathcal{T}(\mathbf{v}_1) + \mathcal{T}(\mathbf{v}_2)$
2. **Homogeneity:** $\mathcal{T}(\alpha \mathbf{v}) = \alpha \mathcal{T}(\mathbf{v})$ for all scalars $\alpha$

However, $\mathrm{L}_\pi$ fails the additivity test:

$$
\mathrm{L}_\pi(\mathbf{v}_1 + \mathbf{v}_2) = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi(\mathbf{v}_1 + \mathbf{v}_2) = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi\mathbf{v}_1 + \gamma \mathbf{P}_\pi\mathbf{v}_2
$$

while

$$
\mathrm{L}_\pi(\mathbf{v}_1) + \mathrm{L}_\pi(\mathbf{v}_2) = (\mathbf{r}_\pi + \gamma \mathbf{P}_\pi\mathbf{v}_1) + (\mathbf{r}_\pi + \gamma \mathbf{P}_\pi\mathbf{v}_2) = 2\mathbf{r}_\pi + \gamma \mathbf{P}_\pi\mathbf{v}_1 + \gamma \mathbf{P}_\pi\mathbf{v}_2.
$$

The presence of the constant term $\mathbf{r}_\pi$ makes $\mathrm{L}_\pi$ affine rather than linear. An affine operator has the form $\mathcal{A}(\mathbf{v}) = \mathbf{b} + \mathcal{T}(\mathbf{v})$, where $\mathbf{b}$ is a constant vector and $\mathcal{T}$ is a linear operator. In our case, $\mathbf{b} = \mathbf{r}_\pi$ and $\mathcal{T}(\mathbf{v}) = \gamma \mathbf{P}_\pi\mathbf{v}$.

Despite this technical distinction, the term "linear operator" is commonly used in the reinforcement learning literature when referring to $\mathrm{L}_\pi$, following a slight abuse of terminology.
```

Therefore, we view $\mathrm{L}_\pi$ as an operator mapping elements of $V$ to $V$: i.e., $\mathrm{L}_\pi: V \rightarrow V$. The fact that the value function of a policy is the solution to a fixed-point equation can then be expressed with the statement: 

$$
\mathbf{v}_\gamma^{\pi}=\mathrm{L}_\pi \mathbf{v}_\gamma^{\pi}.
$$

This is a **fixed-point equation**: the value function $\mathbf{v}_\gamma^{\pi}$ is a fixed point of the operator $\mathrm{L}_\pi$.

## Solving Operator Equations

The operator equation we encountered in MDPs, $\mathbf{v}_\gamma^{\pi} = \mathrm{L}_\pi \mathbf{v}_\gamma^{\pi}$, is a specific instance of a more general class of problems known as operator equations. These equations appear in various fields of mathematics and applied sciences, ranging from differential equations to functional analysis.

Operator equations can take several forms, each with its own characteristics and solution methods:

1. **Fixed Point Form**: $x = \mathrm{T}(x)$, where $\mathrm{T}: X \rightarrow X$.
   Common in fixed-point problems, such as our MDP equation, we seek a fixed point $x^*$ such that $x^* = \mathrm{T}(x^*)$.

2. **General Operator Equation**: $\mathrm{T}(x) = y$, where $\mathrm{T}: X \rightarrow Y$.
   Here, $X$ and $Y$ can be different spaces. We seek an $x \in X$ that satisfies the equation for a given $y \in Y$.

3. **Nonlinear Equation**: $\mathrm{T}(x) = 0$, where $\mathrm{T}: X \rightarrow Y$.
   A special case of the general operator equation where we seek roots or zeros of the operator.

4. **Variational Inequality**: Find $x^* \in K$ such that $\langle \mathrm{T}(x^*), x - x^* \rangle \geq 0$ for all $x \in K$.
   Here, $K$ is a closed convex subset of $X$, and $\mathrm{T}: K \rightarrow X^*$ (the dual space of $X$). These problems often arise in optimization, game theory, and partial differential equations.

### Successive Approximation Method

For equations in fixed point form, a common numerical solution method is successive approximation, also known as fixed-point iteration:

````{prf:algorithm} Successive Approximation
:label: successive-approximation

**Input:** An operator $\mathrm{T}: X \rightarrow X$, an initial guess $x_0 \in X$, and a tolerance $\epsilon > 0$  
**Output:** An approximate fixed point $x^*$ such that $\|x^* - \mathrm{T}(x^*)\| < \epsilon$

1. Initialize $n = 0$  
2. **repeat**  
    3. Compute $x_{n+1} = \mathrm{T}(x_n)$  
    4. If $\|x_{n+1} - x_n\| < \epsilon$, **return** $x_{n+1}$  
    5. Set $n = n + 1$  
6. **until** convergence or maximum iterations reached  

````

The convergence of successive approximation depends on the properties of the operator $\mathrm{T}$. In the simplest and most common setting, we assume $\mathrm{T}$ is a contraction mapping. The Banach Fixed-Point Theorem then guarantees that $\mathrm{T}$ has a unique fixed point, and the successive approximation method will converge to this fixed point from any starting point. Specifically, $\mathrm{T}$ is a contraction if there exists a constant $q \in [0,1)$ such that for all $x,y \in X$:

$$
d(\mathrm{T}(x), \mathrm{T}(y)) \leq q \cdot d(x,y)
$$

where $d$ is the metric on $X$. In this case, the rate of convergence is linear, with error bound:

$$
d(x_n, x^*) \leq \frac{q^n}{1-q} d(x_1, x_0)
$$

However, the contraction mapping condition is not the only one that can lead to convergence. For instance, if $\mathrm{T}$ is nonexpansive (i.e., Lipschitz continuous with Lipschitz constant 1) and $X$ is a Banach space with certain geometrical properties (e.g., uniformly convex), then under additional conditions (e.g., $\mathrm{T}$ has at least one fixed point), the successive approximation method can still converge, albeit potentially more slowly than in the contraction case.

In practice, when dealing with specific problems like MDPs or differential equations, the properties of the operator often naturally align with one of these convergence conditions. For example, in discounted MDPs, the Bellman operator is a contraction in the supremum norm, which guarantees the convergence of value iteration.

### Newton-Kantorovich Method

The Newton-Kantorovich method is a generalization of Newton's method from finite dimensional vector spaces to infinite dimensional function spaces: rather than iterating in the space of vectors, we are iterating in the space of functions. 

Newton's method is often written as the familiar update:

$$
x_{k+1} = x_k - [DF(x_k)]^{-1} F(x_k),
$$
which makes it look as though the essence of the method is "take a derivative and invert it." But the real workhorse behind Newton's method (both in finite and infinite dimensions) is **linearization**.

At each step, the idea is to replace the nonlinear operator $F:X \to Y$ by a local surrogate model of the form

$$
F(x+h) \approx F(x) + Lh,
$$
where $L$ is a linear map capturing how small perturbations in the input propagate to changes in the output. This is a Taylor-like expansion in Banach spaces: the role of the derivative is precisely to provide the correct notion of such a linear operator.

To find a root of $F$, we impose the condition that the surrogate vanishes at the next iterate:

$$
0 = F(x+h) \approx F(x) + Lh.
$$
Solving this linear equation gives the increment $h$. In finite dimensions, $L$ is the Jacobian matrix; in Banach spaces, it must be the **Fréchet derivative**.

But what exactly is a Fréchet derivative in infinite dimensions? To understand this, we need to generalize the concept of derivative from finite-dimensional calculus. In infinite-dimensional spaces, there are several notions of differentiability, each with different strengths and requirements:

**1. Gâteaux (Directional) Derivative**

We say that the Gâteaux derivative of $F$ at $x$ in a specific direction $h$ is:

$$
F'(x; h) = \lim_{t \to 0} \frac{F(x + th) - F(x)}{t}
$$

This quantity measures how the function $F$ changes along the ray $x + th$. While this limit may exist for each direction $h$ separately, it doesn't guarantee that the derivative is linear in $h$. This is a key limitation: the Gâteaux derivative can exist in all directions but still fail to provide a good linear approximation.

**2. Hadamard Directional Derivative**

Rather than considering a single direction of perturbation, we now consider a bundle of perturbations around $h$. We ask how the function changes as we approach the target direction from nearby directions. We say that $F$ has a Hadamard directional derivative if:

$$
F'(x; h) = \lim_{\substack{t \downarrow 0 \\ h' \to h}} \frac{F(x + t h') - F(x)}{t}
$$

This is a stronger condition than Gâteaux differentiability because it requires the limit to be uniform over nearby directions. However, it still doesn't guarantee linearity in $h$.

**3. Fréchet Derivative**

The strongest and most natural notion: $F$ is Fréchet differentiable at $x$ if there exists a bounded linear operator $L$ such that:

$$
\lim_{h \to 0} \frac{\|F(x + h) - F(x) - Lh\|}{\|h\|} = 0
$$

This definition directly addresses the inadequacy of the previous notions. Unlike Gâteaux and Hadamard derivatives, the Fréchet derivative explicitly requires the existence of a linear operator $L$ that provides a good approximation. Key properties:

- $L$ must be **linear** in $h$ (unlike the directional derivatives above)
- The approximation error is $o(\|h\|)$, uniform in all directions
- This is the "true" derivative: it generalizes the Jacobian matrix to infinite dimensions
- Notation: $L = F'(x)$ or $DF(x)$

**Relationship:**

$$
\text{Fréchet differentiable} \Rightarrow \text{Hadamard directionally diff.} \Rightarrow \text{Gâteaux directionally diff.}
$$

In the context of the Newton-Kantorovich method, we work with an operator $F: X \to Y$ where both $X$ and $Y$ are Banach spaces. The Fréchet derivative $F'(x)$ is the best linear approximation of $F$ near $x$, and it's exactly this linear operator $L$ that we use in our linearization $F(x+h) \approx F(x) + F'(x)h$.

Now apart from those mathematical technicalities, Newton-Kantorovich has in essence the same structure as that of the original Newton's method. That is, it applies the following sequence of steps:

1. **Linearize the Operator**:
   Given an approximation $ x_n $, we consider the Fréchet derivative of $ F $, denoted by $ F'(x_n) $. This derivative is a linear operator that provides a local approximation of $ F $ near $ x_n $.

2. **Set Up the Newton Step**:
   The method then solves the linearized equation for a correction $ h_n $:

   $$
   F'(x_n) h_n = -F(x_n).
   $$
   This equation represents a linear system where $ h_n $ is chosen so that the linearized operator $ F(x_n) + F'(x_n)h_n $ equals zero.

3. **Update the Solution**:
   The new approximation $ x_{n+1} $ is then given by:

   $$
   x_{n+1} = x_n + h_n.
   $$
   This correction step refines $ x_n $, bringing it closer to the true solution.

4. **Repeat Until Convergence**:
   We repeat the linearization and update steps until the solution $ x_n $ converges to the desired tolerance, which can be verified by checking that $ \|F(x_n)\| $ is sufficiently small, or by monitoring the norm $ \|x_{n+1} - x_n\| $.

The convergence of Newton-Kantorovich does not hinge on $ F $ being a contraction over the entire domain (as it could be the case for successive approximation). The convergence properties of the Newton-Kantorovich method are as follows:

1. **Local Convergence**: Under mild conditions (e.g., $F$ is Fréchet differentiable and $F'(x)$ is invertible near the solution), the method converges locally. This means that if the initial guess is sufficiently close to the true solution, the method will converge.

2. **Global Convergence**: Global convergence is not guaranteed in general. However, under stronger conditions (e.g., $F$ is analytic and satisfies certain bounds), the method can converge globally.

3. **Rate of Convergence**: When the method converges, it typically exhibits quadratic convergence. This means that the error at each step is proportional to the square of the error at the previous step:

   $$
   \|x_{n+1} - x^*\| \leq C\|x_n - x^*\|^2
   $$

   where $x^*$ is the true solution and $C$ is some constant. This quadratic convergence is significantly faster than the linear convergence typically seen in methods like successive approximation.

## Optimality Equations for Infinite-Horizon MDPs

Recall that in the finite-horizon setting, the optimality equations are:

$$
v_n(s) = \max_{a \in A_s} \left\{r(s, a) + \gamma \sum_{j \in S} p(j | s, a) v_{n+1}(j)\right\}
$$

where $v_n(s)$ is the value function at time step $n$ for state $s$, $A_s$ is the set of actions available in state $s$, $r(s, a)$ is the reward function, $\gamma$ is the discount factor, and $p(j | s, a)$ is the transition probability from state $s$ to state $j$ given action $a$.

Intuitively, we would expect that by taking the limit of $n$ to infinity, we might get the nonlinear equations:

$$
v(s) = \max_{a \in A_s} \left\{r(s, a) + \gamma \sum_{j \in S} p(j | s, a) v(j)\right\}
$$

which are called the optimality equations or Bellman equations for infinite-horizon MDPs.

We can adopt an operator-theoretic perspective by defining operators on the space $V$ of bounded real-valued functions on the state space $S$. For a deterministic Markov rule $\pi \in \Pi^{MD}$, define the **policy-evaluation operator**:

$$
(\BellmanPi v)(s) = r(s,\pi(s)) + \gamma \sum_{j \in \mathcal{S}} p(j|s,\pi(s)) v(j)
$$

The **Bellman optimality operator** is then:

$$
\Bellman \mathbf{v} \equiv \max_{\pi \in \Pi^{MD}} \left\{\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}\right\}
$$

where $\Pi^{MD}$ is the set of Markov deterministic decision rules, $\mathbf{r}_\pi$ is the reward vector under decision rule $\pi$, and $\mathbf{P}_\pi$ is the transition probability matrix under decision rule $\pi$.

Note that while we write $\max_{\pi \in \Pi^{MD}}$, we do not implement the above operator by enumerating all decision rules. Rather, the fact that we compare policies based on their value functions in a componentwise fashion means that maximizing over the space of Markovian deterministic rules reduces to the following update in component form:

$$
(\Bellman \mathbf{v})(s) = \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right\}
$$

For convenience, we define the **greedy selector** $\mathrm{Greedy}(v) \in \Pi^{MD}$ that extracts an optimal decision rule from a value function:

$$
\mathrm{Greedy}(v)(s) \in \arg\max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right\}
$$

In Puterman's terminology, such a greedy selector is called **$v$-improving** (or **conserving** when it achieves the maximum). This operator will be useful for expressing algorithms succinctly:
- **Value iteration:** $v_{k+1} = \Bellman v_k$, then extract $\pi = \mathrm{Greedy}(v^*)$
- **Policy iteration:** $\pi_{k+1} = \mathrm{Greedy}(v^{\pi_k})$ with $v^{\pi_k}$ solving $v = \mathrm{L}_{\pi_k}v$

The equivalence between these two forms can be shown mathematically, as demonstrated in the following proposition and proof.

```{prf:proposition}
The operator $\Bellman$ defined as a maximization over Markov deterministic decision rules:

$$(\Bellman \mathbf{v})(s) = \max_{\pi \in \Pi^{MD}} \left\{r(s,\pi(s)) + \gamma \sum_{j \in \mathcal{S}} p(j|s,\pi(s)) v(j)\right\}$$

is equivalent to the componentwise maximization over actions:

$$(\Bellman \mathbf{v})(s) = \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right\}$$
```

```{prf:proof}
Fix $s$. Let 

$$
Q_v(s,a) \triangleq r(s,a)+\gamma\sum_{j}p(j\mid s,a)\,v(j).
$$

For any rule $\pi \in \Pi^{MD}$, we have $(\BellmanPi v)(s)=Q_v(s,\pi(s))\le \max_{a\in\mathcal{A}_s}Q_v(s,a)$.

Taking the maximum over $\pi$ gives

$$
\max_{\pi\in\Pi^{MD}}(\BellmanPi v)(s) \le \max_{a\in\mathcal{A}_s}Q_v(s,a).
$$

Conversely, choose a **greedy selector** $\pi^v\in\Pi^{MD}$ such that for each $s$,

$$\pi^v(s)\in\arg\max_{a\in\mathcal{A}_s}Q_v(s,a)$$

(possible since $\mathcal{A}_s$ is finite; otherwise use a measurable $\varepsilon$-greedy selector). Then

$$
(\Bellman _{\pi^v}v)(s)=Q_v(s,\pi^v(s))=\max_{a\in\mathcal{A}_s}Q_v(s,a),
$$

so $\max_{\pi}(\BellmanPi v)(s)\ge \max_{a}Q_v(s,a)$. Combining both inequalities yields equality.
```

## Algorithms for Solving the Optimality Equations

The optimality equations are operator equations. Therefore, we can apply general numerical methods to solve them. Applying the successive approximation method to the Bellman optimality equation yields a method known as "value iteration" in dynamic programming. A direct application of the blueprint for successive approximation yields the following algorithm:

````{prf:algorithm} Value Iteration
:label: value-iteration

**Input** Given an MDP $(S, A, P, R, \gamma)$ and tolerance $\varepsilon > 0$  

**Output** Compute an $\varepsilon$-optimal value function $v$ and policy $\pi$  

1. Initialize $v_0(s) = 0$ for all $s \in S$  
2. $n \leftarrow 0$  
3. **repeat**  

    1. For each $s \in S$:  

        1. $v_{n+1}(s) \leftarrow (\Bellman v_n)(s) = \max_{a \in A} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)v_n(j)\right\}$  

    2. $\delta \leftarrow \|v_{n+1} - v_n\|_\infty$  
    3. $n \leftarrow n + 1$  

4. **until** $\delta < \frac{\varepsilon(1-\gamma)}{2\gamma}$  
5. Extract greedy policy: $\pi \leftarrow \mathrm{Greedy}(v_n)$ where

    $$\mathrm{Greedy}(v)(s) \in \arg\max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)v(j)\right\}$$

6. **return** $v_n, \pi$  
````

The termination criterion in this algorithm is based on a specific bound that provides guarantees on the quality of the solution. This is in contrast to supervised learning, where we often use arbitrary termination criteria based on computational budget or early stopping when the learning curve flattens. This is because establishing implementable generalization bounds in supervised learning is challenging.

However, in the dynamic programming context, we can derive various bounds that can be implemented in practice. These bounds help us terminate our procedure with a guarantee on the precision of our value function and, correspondingly, on the optimality of the resulting policy.

````{prf:proposition} Convergence of Value Iteration 
:label: value-iteration-convergence
(Adapted from {cite:t}`Puterman1994` theorem 6.3.1)

Let $v_0$ be any initial value function, $\varepsilon > 0$ a desired accuracy, and let $\{v_n\}$ be the sequence of value functions generated by value iteration, i.e., $v_{n+1} = \Bellman v_n$ for $n \geq 0$, where $\Bellman$ is the Bellman optimality operator. Then:

1. $v_n$ converges to the optimal value function $v^*_\gamma$,
2. The algorithm terminates in finite time,
3. The resulting policy $\pi_\varepsilon$ is $\varepsilon$-optimal, and
4. When the algorithm terminates, $v_{n+1}$ is within $\varepsilon/2$ of $v^*_\gamma$.

````

````{prf:proof}
Parts 1 and 2 follow directly from the fact that $\Bellman$ is a contraction mapping. Hence, by Banach's fixed-point theorem, it has a unique fixed point (which is $v^*_\gamma$), and repeated application of $\Bellman$ will converge to this fixed point. Moreover, this convergence happens at a geometric rate, which ensures that we reach the termination condition in finite time.

To show that the Bellman optimality operator $\Bellman$ is a contraction mapping, we need to prove that for any two value functions $v$ and $u$:

$$\|\Bellman v - \Bellman u\|_\infty \leq \gamma \|v - u\|_\infty$$

where $\gamma \in [0,1)$ is the discount factor and $\|\cdot\|_\infty$ is the supremum norm.

Let's start by writing out the definition of $\Bellman v$ and $\Bellman u$:

$$\begin{align*}
(\Bellman v)(s) &= \max_{a \in A} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)v(j)\right\}\\
(\Bellman u)(s) &= \max_{a \in A} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)u(j)\right\}
\end{align*}$$

For any state $s$, let $a_v$ be the action that achieves the maximum for $(\Bellman v)(s)$, and $a_u$ be the action that achieves the maximum for $(\Bellman u)(s)$. By the definition of these maximizers:

$$\begin{align*}
(\Bellman v)(s) &\geq r(s,a_u) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a_u)v(j)\\
(\Bellman u)(s) &\geq r(s,a_v) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a_v)u(j)
\end{align*}$$

Subtracting these inequalities:

$$\begin{align*}
(\Bellman v)(s) - (\Bellman u)(s) &\leq \gamma \sum_{j \in \mathcal{S}} p(j|s,a_v)(v(j) - u(j))\\
(\Bellman u)(s) - (\Bellman v)(s) &\leq \gamma \sum_{j \in \mathcal{S}} p(j|s,a_u)(u(j) - v(j))
\end{align*}$$

Taking the absolute value and using the fact that $\sum_{j \in \mathcal{S}} p(j|s,a) = 1$:

$$|(\Bellman v)(s) - (\Bellman u)(s)| \leq \gamma \max_{j \in \mathcal{S}} |v(j) - u(j)| = \gamma \|v - u\|_\infty$$

Since this holds for all $s \in \mathcal{S}$, taking the supremum over $s$ gives:

$$\|\Bellman v - \Bellman u\|_\infty \leq \gamma \|v - u\|_\infty$$

Thus, $\Bellman$ is a contraction mapping with contraction factor $\gamma$.

Now, let's prove parts 3 and 4. Suppose the algorithm has just terminated, i.e., $\|v_{n+1} - v_n\|_\infty < \frac{\varepsilon(1-\gamma)}{2\gamma}$ for some $n$. We want to show that our current value function $v_{n+1}$ and the policy $\pi_\varepsilon$ derived from it are close to optimal.

By the triangle inequality:

$$\|v^{\pi_\varepsilon}_\gamma - v^*_\gamma\|_\infty \leq \|v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty + \|v_{n+1} - v^*_\gamma\|_\infty$$

For the first term, since $v^{\pi_\varepsilon}_\gamma$ is the fixed point of $\mathrm{L}_{\pi_\varepsilon}$ and $\pi_\varepsilon$ is greedy with respect to $v_{n+1}$ (i.e., $\mathrm{L}_{\pi_\varepsilon}v_{n+1} = \Bellman v_{n+1}$):

$$
\begin{aligned}
\|v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty &= \|\mathrm{L}_{\pi_\varepsilon}v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty \\
&\leq \|\mathrm{L}_{\pi_\varepsilon}v^{\pi_\varepsilon}_\gamma - \mathrm{L}_{\pi_\varepsilon}v_{n+1}\|_\infty + \|\mathrm{L}_{\pi_\varepsilon}v_{n+1} - v_{n+1}\|_\infty \\
&= \|\mathrm{L}_{\pi_\varepsilon}v^{\pi_\varepsilon}_\gamma - \mathrm{L}_{\pi_\varepsilon}v_{n+1}\|_\infty + \|\Bellman v_{n+1} - v_{n+1}\|_\infty \\
&\leq \gamma\|v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty + \gamma\|v_{n+1} - v_n\|_\infty
\end{aligned}
$$

where we used that both $\Bellman$ and $\mathrm{L}_{\pi_\varepsilon}$ are contractions with factor $\gamma$, and that $v_{n+1} = \Bellman v_n$.

Rearranging:

$$\|v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty \leq \frac{\gamma}{1-\gamma}\|v_{n+1} - v_n\|_\infty$$

Similarly, since $v^*_\gamma$ is the fixed point of $\Bellman$:

$$\|v_{n+1} - v^*_\gamma\|_\infty = \|\Bellman v_n - \Bellman v^*_\gamma\|_\infty \leq \gamma\|v_n - v^*_\gamma\|_\infty \leq \frac{\gamma}{1-\gamma}\|v_{n+1} - v_n\|_\infty$$

Since $\|v_{n+1} - v_n\|_\infty < \frac{\varepsilon(1-\gamma)}{2\gamma}$:

$$\|v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty \leq \frac{\gamma}{1-\gamma} \cdot \frac{\varepsilon(1-\gamma)}{2\gamma} = \frac{\varepsilon}{2}$$

$$\|v_{n+1} - v^*_\gamma\|_\infty \leq \frac{\gamma}{1-\gamma} \cdot \frac{\varepsilon(1-\gamma)}{2\gamma} = \frac{\varepsilon}{2}$$

Combining these:

$$\|v^{\pi_\varepsilon}_\gamma - v^*_\gamma\|_\infty \leq \frac{\varepsilon}{2} + \frac{\varepsilon}{2} = \varepsilon$$

This completes the proof, showing that $v_{n+1}$ is within $\varepsilon/2$ of $v^*_\gamma$ (part 4) and $\pi_\varepsilon$ is $\varepsilon$-optimal (part 3).
````

### Bellman contraction laboratory

Use the controls below to change the discount factor, transition persistence, reward asymmetry, and starting value. Before moving $\gamma$, predict how it will change the slope of the error envelope. The middle panel compares the observed error and Bellman residual with the contraction bound; the text below reports the final greedy policy.

```{marimo-config}
:echo: false
:error: false
:pyproject:

requires-python = ">=3.12"
dependencies = ["matplotlib", "numpy"]
```

```{marimo} python
import marimo as mo
import matplotlib.pyplot as plt
import numpy as np

discount = mo.ui.slider(start=0.10, stop=0.99, step=0.01, value=0.90, label="Discount γ")
persistence = mo.ui.slider(start=0.50, stop=0.99, step=0.01, value=0.85, label="Transition persistence")
reward_asymmetry = mo.ui.slider(start=-2.0, stop=2.0, step=0.1, value=0.8, label="Reward asymmetry")
initial_value = mo.ui.slider(start=-10.0, stop=10.0, step=0.5, value=0.0, label="Initial value scale")
mo.vstack([discount, persistence, reward_asymmetry, initial_value])
```

```{marimo} python
gamma_lab = discount.value
p_lab = persistence.value
reward_gap_lab = reward_asymmetry.value

transitions_lab = np.array([
    [[p_lab, 1 - p_lab], [1 - p_lab, p_lab]],
    [[1 - p_lab, p_lab], [p_lab, 1 - p_lab]],
])
rewards_lab = np.array([
    [1.0 + reward_gap_lab, 0.0],
    [0.0, 1.0 - reward_gap_lab],
])

def bellman_lab(value_lab):
    q_lab = rewards_lab + gamma_lab * np.einsum("asj,j->sa", transitions_lab, value_lab)
    return q_lab.max(axis=1), q_lab

v_star_lab = np.zeros(2)
for _ in range(1000):
    next_star_lab, _ = bellman_lab(v_star_lab)
    if np.max(np.abs(next_star_lab - v_star_lab)) < 1e-12:
        break
    v_star_lab = next_star_lab

value_lab = np.array([initial_value.value, -initial_value.value], dtype=float)
trace_lab = [value_lab.copy()]
residual_lab = []
error_lab = [np.max(np.abs(value_lab - v_star_lab))]
for _ in range(30):
    next_value_lab, _ = bellman_lab(value_lab)
    residual_lab.append(np.max(np.abs(next_value_lab - value_lab)))
    value_lab = next_value_lab
    trace_lab.append(value_lab.copy())
    error_lab.append(np.max(np.abs(value_lab - v_star_lab)))

trace_lab = np.asarray(trace_lab)
error_lab = np.asarray(error_lab)
residual_lab = np.asarray(residual_lab)
bound_lab = error_lab[0] * gamma_lab ** np.arange(error_lab.size)
_, final_q_lab = bellman_lab(value_lab)
policy_lab = final_q_lab.argmax(axis=1)
```

```{marimo} python
fig_lab, axes_lab = plt.subplots(1, 2, figsize=(10, 3.6))
axes_lab[0].plot(trace_lab[:, 0], label="state 0")
axes_lab[0].plot(trace_lab[:, 1], label="state 1")
axes_lab[0].axhline(v_star_lab[0], color="C0", linestyle=":", alpha=0.7)
axes_lab[0].axhline(v_star_lab[1], color="C1", linestyle=":", alpha=0.7)
axes_lab[0].set(xlabel="Iteration", ylabel="Value", title="Value-iteration trace")
axes_lab[0].legend()

axes_lab[1].semilogy(error_lab, label="actual error")
axes_lab[1].semilogy(bound_lab, linestyle="--", label="contraction bound")
axes_lab[1].semilogy(np.arange(1, residual_lab.size + 1), residual_lab, linestyle=":", label="Bellman residual")
axes_lab[1].set(xlabel="Iteration", ylabel="Sup norm", title="Error certificate")
axes_lab[1].legend()
fig_lab.tight_layout()

mo.vstack([
    fig_lab,
    mo.md(
        f"**Greedy policy:** state 0 → action {policy_lab[0]}, "
        f"state 1 → action {policy_lab[1]}.  "
        f"Final residual: **{residual_lab[-1]:.2e}**; "
        f"final error: **{error_lab[-1]:.2e}**."
    ),
])
```

:::{figure} _static/bellman-contraction-fallback.png
:label: fig-bellman-contraction-fallback
:class: pdf-fallback
:alt: Static Bellman contraction laboratory preview

Static preview of value-iteration traces and a geometric contraction bound. The online book provides controls for $\gamma$, transitions, rewards, and the initial value.
:::

## Exact Scheduling MDP for Inference Serving

The inference examples have so far treated the scheduling rule as fixed and the
GPU clock as the action. A different decision interface fixes the clock and
asks which phase should receive the next unit of service. Prefill admits new
requests into decode and consumes cache; decode advances requests already
producing output tokens. Serving either phase delays the other.

An exact request-level Markov state would contain every prompt length, generated
token count, cache allocation, and waiting time. For computation, these
quantities are aggregated into the finite state

$$
s=(p,d,a)\in\{0,\ldots,6\}^2\times\{0,\ldots,4\},
$$

where $p$ counts waiting prefill jobs, $d$ counts active decode jobs, and $a$
is the oldest prefill-age bin. The actions are

$$
\mathcal A=\{\text{prefill},\text{decode},\text{idle}\}.
$$

An action is masked when its phase is empty. Prefill is also masked at $d=6$,
which represents the cache limit in this abstraction. The empirical arrival
rate comes from the same Azure trace used in the modeling chapter and defines a
Bernoulli arrival probability for each 0.1-second decision period. A prefill
action completes one aggregate prompt with a Bernoulli probability derived
from the median-clock prefill rate and the trace's mean prompt length. A
successful completion moves that job into decode. During a decode action, each
active job completes independently with a probability derived from the
median-clock decode rate and the trace's mean output length. Queue counts are
capped at six, and arrivals beyond that cap are recorded as drops. These
choices define a complete transition matrix $P_{ss'}^u$ on 245 states.

The one-step cost assigns separate penalties to congestion, old prompt work,
decode stalls, dropped requests, and energy:

$$
c(s,u)=p+d+4\mathbf 1\{a=4\}
+2d\mathbf 1\{u\ne\text{decode}\}
+10\mathbb E[N_{\mathrm{drop}}\mid s,u]
+0.1\frac{E(u)}{E_{\max}}.
$$

For prefill, decode, and idle, respectively, the committed profile gives
$E(u)=(5.0,4.4,1.9)$ joules per decision period at the median clock. Only the
ratio $E(u)/E_{\max}$ enters the stage cost. These values inherit the profile's
provenance; with the current engineering surrogate, they are not L4
measurements.

With $\gamma=0.99$, cost-minimizing value iteration applies

$$
\begin{aligned}
Q_n(s,u)&=c(s,u)+\gamma\sum_{s'}P_{ss'}^uV_n(s'),\\
V_{n+1}(s)&=\min_{u\in\mathcal A(s)}Q_n(s,u).
\end{aligned}
$$

Iteration stops when $\lVert V_{n+1}-V_n\rVert_\infty<10^{-10}$. The final
Bellman residual is checked independently and must be below $10^{-8}$.

The experiment asks how the optimal phase decision changes with the two queue
lengths and the age of the oldest prompt. Policy slices show the selected
action for each $(p,d)$ pair. A replay then applies the resulting policy to
fixed evaluation episodes and compares it with fixed phase-priority rules.

```{code-cell} python
:tags: [remove-input]
:label: fig-inference-scheduling-dp
:caption: Exact value iteration on the reduced inference-scheduling MDP. Policy slices vary the prefill and decode queue lengths and the oldest prefill-age bin. The replay uses the same fixed-clock transition model and reveals arrivals and actions only as they occur.

from pathlib import Path
import sys

from IPython.display import HTML, display

code_dir = Path.cwd() / "code"
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from inference_replay import render_serving_replay

display(HTML(render_serving_replay(
    Path("artifacts/inference_serving/textbook_results.json"),
    view="scheduling",
)))
```

:::{figure} _static/inference_serving/scheduling.svg
:label: fig-inference-scheduling-dp-fallback
:class: pdf-fallback
:alt: Static policy slices for prefill, decode, and idle actions in the reduced scheduling MDP.

Static policy slices and evaluation replay from the exact scheduling MDP. The
online book adds age selection, playback, stepping, and scrubbing.
:::

```{code-cell} python
:tags: [remove-input]
:label: tbl-inference-dp-metrics
:caption: Value-iteration certificate and transition parameters for the reduced scheduling MDP. The Bellman residual applies only to this finite model.

import pandas as pd

pd.read_csv("artifacts/inference_serving/metrics_dp.csv")
```

Value iteration required 2,480 sweeps on this high-discount problem. Its
independently recomputed Bellman residual is $9.83\times10^{-11}$, below both
the $10^{-8}$ acceptance threshold and the $10^{-10}$ stopping tolerance. The
certificate concerns the supplied 245-state transition matrix.

The result is exact for the stated finite MDP, not for vLLM. Aggregating request
ages and lengths removes distinctions that can affect head-of-line waiting and
cache release. The empirical transition kernel is stationary, the clock is
fixed, and the action set excludes mixed prefill-decode batches. The
request-level replay from the earlier chapters is therefore a model-audit tool,
not part of the optimality proof. Profile provenance affects the empirical
interpretation of the transition probabilities but not the numerical
optimality certificate for the finite matrix supplied to value iteration.

:::{dropdown} Inspect the scheduling MDP and value iteration
```{literalinclude} code/inference_control.py
:language: python
:start-at: def solve_scheduling_mdp
:end-before: def _sample_next_state
:linenos:
```

{download}`Download the complete inference-control implementation <code/inference_control.py>`
:::

### Newton-Kantorovich Applied to Bellman Optimality

We now apply the Newton-Kantorovich framework to the Bellman optimality equation. Let

$$
(\Bellman v)(s) = \max_{a \in A(s)} \left\{ r(s,a) + \gamma \sum_{s'} p(s' \mid s,a) v(s') \right\}.
$$

The problem is to find $v$ such that $\Bellman v = v$, or equivalently $\mathrm{B}(v) := \Bellman v - v = 0$. The operator $\Bellman$ is piecewise affine, hence not globally differentiable, but it is directionally differentiable everywhere in the Hadamard sense and Fréchet differentiable at points where the maximizer is unique.

We consider three complementary perspectives for understanding and computing its derivative.

#### Perspective 1: Max of Affine Maps

In tabular form, for finite state and action spaces, the Bellman operator can be written as a pointwise maximum of affine maps:

$$
(\Bellman v)(s) = \max_{a \in A(s)} \left\{ r(s,a) + \gamma (P_a v)(s) \right\},
$$
where $P_a \in \mathbb{R}^{|S| \times |S|}$ is the transition matrix associated with action $a$. Each $Q_a v := r^a + \gamma P_a v$ is affine in $v$. The operator $\Bellman$ therefore computes the upper envelope of a finite set of affine functions at each state.

At any $v$, let the **active set** at state $s$ be

$$
\mathcal{A}^*(s; v) := \arg\max_{a \in A(s)} (Q_a v)(s).
$$

Then the Hadamard directional derivative exists and is given by

$$
(\Bellman '(v; h))(s) = \max_{a \in \mathcal{A}^*(s; v)} \gamma (P_a h)(s).
$$

If the active set is a singleton, this expression becomes linear in $h$, and $\Bellman$ is Fréchet differentiable at $v$, with

$$
\Bellman'(v) = \gamma P_{\pi_v},
$$

where $\pi_v(s) := a^*(s)$ is the greedy policy at $v$. 
<!-- In the presence of ties, the derivative becomes set-valued: the Clarke subdifferential consists of stochastic matrices whose rows are convex combinations of the $\gamma P_a$ over $a \in \mathcal{A}^*(s; v)$. -->

#### Perspective 2: Envelope Theorem

Consider now a value function approximated as a linear combination of basis functions:

$$
v_c(s) = \sum_j c_j \phi_j(s).
$$

At a node $s_i$, define the parametric maximization

$$
v_i(c) := (\Bellman v_c)(s_i) = \max_{a \in A(s_i)} \left\{ r(s_i,a) + \gamma \sum_j c_j \mathbb{E}_{s' \mid s_i, a}[\phi_j(s')] \right\}.
$$

Define

$$
F_i(a, c) := r(s_i,a) + \gamma \sum_j c_j \mathbb{E}_{s' \mid s_i, a}[\phi_j(s')],
$$

so that $v_i(c) = \max_a F_i(a, c)$. Since $F_i$ is linear in $c$, we can apply the **envelope theorem** (Danskin's theorem): if the optimizer $a_i^*(c)$ is unique or selected measurably, then

$$
\frac{\partial v_i}{\partial c_j}(c) = \gamma \mathbb{E}_{s' \mid s_i, a_i^*(c)}[\phi_j(s')].
$$

We do not need to differentiate the optimizer $a_i^*(c)$ itself. The result extends to the subdifferential case when ties occur, where the Jacobian becomes set-valued.

This result is useful when solving the collocation equation $\Phi c = v(c)$. Newton's method requires the Jacobian $v'(c)$, and this expression allows us to compute it without involving any derivatives of the optimal action.

#### Perspective 3: The Implicit Function Theorem

The third perspective applies the implicit function theorem to understand when the Bellman operator is differentiable despite containing a max operator. The maximization problem defines an implicit relationship between the value function and the optimal action, and the implicit function theorem tells us when this relationship is smooth enough to differentiate through.

The Bellman operator is defined as

$$
(\Bellman v)(s) = \max_{a} \left\{ r(s,a) + \gamma \sum_j p(j \mid s,a) v(j) \right\}.
$$

The difficulty is that the max operator encodes a discrete selection: which action achieves the maximum. To apply the implicit function theorem, we reformulate this as follows. For each action $a$, define the **action-value function**:

$$
Q_a(v, s) := r(s,a) + \gamma \sum_j p(j \mid s,a) v(j).
$$

The optimal action at $v$ satisfies the **optimality condition**:

$$
Q_{a^*(s)}(v, s) \geq Q_a(v, s) \quad \text{for all } a.
$$

Now suppose that at a particular $v$, action $a^*(s)$ is a **strict local maximizer** in the sense that there exists $\delta > 0$ such that

$$
Q_{a^*(s)}(v, s) > Q_a(v, s) + \delta \quad \text{for all } a \neq a^*(s).
$$

This strict inequality is the regularity condition needed for the implicit function theorem. It ensures that the optimal action is unique at $v$ and remains so in a neighborhood of $v$.

To see why, consider any perturbation $v + h$ with $\|h\|$ small. Since $Q_a$ is linear in $v$, we have:

$$
Q_a(v+h, s) = Q_a(v, s) + \gamma \sum_j p(j \mid s,a) h(j).
$$

The perturbation term is bounded: $|\gamma \sum_j p(j \mid s,a) h(j)| \leq \gamma \|h\|$. Therefore, for $\|h\| < \delta/\gamma$, the strict gap ensures that

$$
Q_{a^*(s)}(v+h, s) > Q_a(v+h, s) \quad \text{for all } a \neq a^*(s).
$$

Thus $a^*(s)$ remains the unique maximizer throughout the neighborhood $\{v + h : \|h\| < \delta/\gamma\}$.

The implicit function theorem now applies: in this neighborhood, the mapping $v \mapsto a^*(s; v)$ is **constant** (and hence smooth), taking the value $a^*(s)$. This allows us to write

$$
(\Bellman v)(s) = Q_{a^*(s)}(v, s) = r(s,a^*(s)) + \gamma \sum_j p(j \mid s,a^*(s)) v(j)
$$
as an explicit formula that holds throughout the neighborhood. Since $Q_{a^*(s)}(\cdot, s)$ is an affine (hence smooth) function of $v$, we can differentiate it:

$$
\frac{d}{dv} (\Bellman v)(s) = \gamma P_{a^*(s)}.
$$

More precisely, for any perturbation $h$:

$$
(\Bellman (v+h))(s) = (\Bellman v)(s) + \gamma \sum_j p(j \mid s,a^*(s)) h(j) + o(\|h\|).
$$

This is the Fréchet derivative:

$$
\Bellman'(v) = \gamma P_{\pi_v},
$$

where $\pi_v(s) = a^*(s)$ is the greedy policy.

**The role of the implicit function theorem**: It guarantees that when the maximizer is unique with a strict gap (the regularity condition), the argmax function $v \mapsto a^*(s; v)$ is locally constant, which removes the non-differentiability of the max operator. Without this regularity condition (specifically, at points where multiple actions tie for optimality), the implicit function theorem does not apply, and the operator is not Fréchet differentiable. The active set perspective (Perspective 1) and the envelope theorem (Perspective 2) provide the tools to handle these non-smooth points.

### Connection to Policy Iteration

We return to the Newton-Kantorovich step:

$$
(I - \Bellman'(v_n)) h_n = v_n - \Bellman v_n,
\quad
v_{n+1} = v_n - h_n.
$$

Suppose $\Bellman'(v_n) = \gamma P_{\pi_{v_n}}$ for the greedy policy $\pi_{v_n}$. Then

$$
(I - \gamma P_{\pi_{v_n}}) v_{n+1} = r^{\pi_{v_n}},
$$

which is exactly policy evaluation for $\pi_{v_n}$. Recomputing the greedy policy from $v_{n+1}$ yields the next iterate.

Thus, **policy iteration is Newton-Kantorovich** applied to the Bellman optimality equation. At points of nondifferentiability (when ties occur), the operator is still semismooth, and policy iteration corresponds to a semismooth Newton method. The envelope theorem is what justifies the simplification of the Jacobian to $\gamma P_{\pi_v}$, bypassing the need to differentiate through the optimizer. This completes the equivalence.

### The Semismooth Newton Perspective

The three perspectives we developed above (the active set view, the envelope theorem, and the implicit function theorem) all point toward a deeper framework for understanding Newton-type methods on non-smooth operators. This framework, known as semismooth Newton methods, was developed precisely to handle operators like the Bellman operator that are piecewise smooth but not globally differentiable. The connection between policy iteration and semismooth Newton methods has been rigorously developed in recent work {cite}`Gargiani2022`.

The classical Newton-Kantorovich method assumes the operator is Fréchet differentiable everywhere. The derivative exists, is unique, and varies continuously with the base point. But the Bellman operator $\Bellman$ violates this assumption at any value function where multiple actions tie for optimality at some state. At such points, the implicit function theorem fails, and there is no unique Fréchet derivative. 

Semismooth Newton methods address this by replacing the notion of a single Jacobian with a generalized derivative that captures the behavior of the operator near non-smooth points. The most commonly used generalized derivative is the Clarke subdifferential, which we can think of as the convex hull of all possible "candidate Jacobians" that arise from limits approaching the non-smooth point from different directions.

For the Bellman residual $\mathrm{B}(v) = \Bellman v - v$, the Clarke subdifferential at a point $v$ can be characterized explicitly using our first perspective. Recall that at each state $s$, we defined the active set $\mathcal{A}^*(s; v) = \arg\max_a Q_a(v, s)$. When this set contains multiple actions, the operator is not Fréchet differentiable. However, it remains directionally differentiable in all directions, and the Clarke subdifferential consists of all matrices of the form

$$
\partial \mathrm{B}(v) = \left\{ I - \gamma P_\pi : \pi(s) \in \mathcal{A}^*(s; v) \text{ for all } s \right\}.
$$

In words, the generalized Jacobian is the set of all matrices $I - \gamma P_\pi$ where $\pi$ is any policy that selects an action from the active set at each state. When the maximizer is unique everywhere, this set reduces to a singleton, and we recover the classical Fréchet derivative. When ties occur, the set has multiple elements: precisely the convex combinations mentioned in Perspective 1.

The semismooth Newton method for solving $\mathrm{B}(v) = 0$ proceeds by selecting an element $J_k \in \partial \mathrm{B}(v_k)$ at each iteration and solving

$$
J_k h_k = -\mathrm{B}(v_k), \quad v_{k+1} = v_k + h_k.
$$

What this tells us is that any choice from the Clarke subdifferential yields a valid Newton-like update. In the context of the Bellman equation, choosing $J_k = I - \gamma P_{\pi_k}$ where $\pi_k$ is any greedy policy corresponds exactly to the policy evaluation step in policy iteration. The freedom in selecting which action to choose when ties occur translates to the freedom in selecting which element of the subdifferential to use.

Under appropriate regularity conditions (specifically, when the residual function is BD-regular or CD-regular), the semismooth Newton method converges locally at a quadratic rate {cite}`Gargiani2022`. This means that near the solution, the error decreases quadratically:
$$
\|v_{k+1} - v^*\| \leq C \|v_k - v^*\|^2.
$$

This theoretical result explains an empirical observation that has long been noted in practice: policy iteration typically converges in very few iterations, often just a handful, even when the state and action spaces are enormous and the space of possible policies is exponentially large. 

The semismooth Newton framework also suggests a spectrum of methods interpolating between value iteration and policy iteration. Value iteration can be interpreted as a Newton-like method where we choose $J_k = I$ at every iteration, ignoring the dependence of $\Bellman$ on $v$ entirely. This choice guarantees global convergence through the contraction property but sacrifices the quadratic local convergence rate. Policy iteration, at the other extreme, uses the full generalized Jacobian $J_k = I - \gamma P_{\pi_k}$, achieving quadratic convergence but at the cost of solving a linear system at each iteration.

Between these extremes lie methods that use approximate Jacobians. One natural variant is to choose $J_k = \alpha I$ for some scalar $\alpha > 1$. This leads to the update

$$
v_{k+1} = \frac{\alpha - 1}{\alpha} v_k + \frac{1}{\alpha} \Bellman v_k.
$$

This is known as $\alpha$-value iteration or successive over-relaxation when $\alpha > 1$. For appropriate choices of $\alpha$, this method retains global convergence while achieving better local rates than standard value iteration, and it requires only pointwise operations rather than solving a linear system. The Newton perspective thus unifies existing algorithms and generates new ones by systematically exploring different approximations to the generalized Jacobian.

The connection to semismooth Newton methods places policy iteration within a broader mathematical framework that extends far beyond dynamic programming. Semismooth Newton methods are used in optimization (for complementarity problems and variational inequalities), in PDE-constrained optimization (for problems with control constraints), and in economics (for equilibrium problems). The Bellman equation, viewed through this lens, is simply one instance of a piecewise smooth equation, and the tools developed for such equations apply directly.

### Policy Iteration 

While we derived policy iteration-like steps from the Newton-Kantorovich method, it's worth examining policy iteration as a standalone algorithm, as it has been traditionally presented in the field of dynamic programming.

The policy iteration algorithm for discounted Markov decision problems is as follows:

````{prf:algorithm} Policy Iteration
:label: policy-iteration-standard

**Input:** MDP $(S, A, P, R, \gamma)$
**Output:** Optimal policy $\pi^*$

1. Initialize: $n = 0$, select an arbitrary decision rule $\pi_0 \in \Pi^{MD}$
2. **repeat**
   3. (Policy evaluation) Obtain $\mathbf{v}^n$ by solving:
   
      $$(\mathbf{I}-\gamma \mathbf{P}_{\pi_n}) \mathbf{v} = \mathbf{r}_{\pi_n}$$

   4. (Policy improvement) Choose $\pi_{n+1} = \mathrm{Greedy}(\mathbf{v}^n)$ where:

       $$\pi_{n+1} \in \arg\max_{\pi \in \Pi^{MD}}\left\{\mathbf{r}_\pi+\gamma \mathbf{P}_\pi \mathbf{v}^n\right\}$$
       
       equivalently, for each $s$:
       
       $$\pi_{n+1}(s) \in \arg\max_{a \in \mathcal{A}_s}\left\{r(s,a)+\gamma \sum_j p(j|s,a) \mathbf{v}^n(j)\right\}$$
       
       Set $\pi_{n+1} = \pi_n$ if possible.

   5. If $\pi_{n+1} = \pi_n$, **return** $\pi^* = \pi_n$

   6. $n = n + 1$
7. **until** convergence
````

As opposed to value iteration, this algorithm produces a sequence of both deterministic Markovian decision rules $\{\pi_n\}$ and value functions $\{\mathbf{v}^n\}$. We recognize in this algorithm the linearization step of the Newton-Kantorovich procedure, which takes place here in the policy evaluation step 3 where we solve the linear system $(\mathbf{I}-\gamma \mathbf{P}_{\pi_n}) \mathbf{v} = \mathbf{r}_{\pi_n}$. In practice, this linear system could be solved either using direct methods (eg. Gaussian elimination), using simple iterative methods such as the successive approximation method for policy evaluation, or more sophisticated methods such as GMRES.

## Self-checks

:::{exercise} Contraction factor
:label: ex-dp-check-1

If two value functions differ by at most $\varepsilon$ in sup norm, by at most how much can their discounted Bellman updates differ?
:::

:::{solution} ex-dp-check-1
:class: dropdown

At most $\gamma\varepsilon$. The discounted Bellman operator is a $\gamma$-contraction in the sup norm.
:::

:::{exercise} Residual certificate
:label: ex-dp-check-2

Value iteration produces a Bellman residual $\|Tv-v\|_\infty=0.02$ with $\gamma=0.9$. Give the standard upper bound on $\|v-v^*\|_\infty$.
:::

:::{solution} ex-dp-check-2
:class: dropdown

$\|v-v^*\|_\infty\leq \|Tv-v\|_\infty/(1-\gamma)=0.02/0.1=0.2$.
:::

:::{exercise} Evaluation versus improvement
:label: ex-dp-check-3

Which step of policy iteration requires solving a linear fixed-policy problem, and which step takes an actionwise maximum?
:::

:::{solution} ex-dp-check-3
:class: dropdown

Policy evaluation solves $(I-\gamma P_\pi)v=r_\pi$. Policy improvement computes action values from that $v$ and chooses a greedy action in each state.
:::

:::{exercise} Reduced scheduling state
:label: ex-dp-check-4

Name one pair of request-level serving states that map to the same reduced
state $(p,d,a)$ but can have different future completion distributions.
:::

:::{solution} ex-dp-check-4
:class: dropdown

Two states can have the same numbers of waiting and decoding requests and the
same oldest prefill-age bin while their active requests have different
remaining output lengths. The reduced state discards those lengths, so its
transition kernel averages over them.
:::
