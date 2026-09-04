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
# Finite-Horizon Dynamic Programming

MPC creates feedback by solving a new trajectory problem from each measured
state. Can the decisions for an entire family of states be computed in advance
instead? Dynamic programming decomposes a finite-horizon control problem into
cost-to-go functions indexed by time and state.

The deterministic discrete-time optimal control problem provides the starting
point. Stochastic successors will enter in the next chapter through conditional
expectations.

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

How does the terminal cost propagate backward into an optimal action and value
for every earlier state?

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

When the next state falls between stored grid points, which interpolation rule
supplies its continuation value without destroying the recursion?

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

What form do the value function and feedback law take when the dynamics are
linear and every cost is quadratic?

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

## Summary and Outlook

Backward recursion replaces one trajectory optimization with a sequence of
state-indexed subproblems. Interpolation extends the recursion beyond a finite
state grid, and the linear-quadratic case reduces the value function to a
Riccati recursion and the policy to linear state feedback.

These recursions still assign one successor to each state and action. How do
the value and policy change when a decision must account for a distribution of
possible successors? [Stochastic dynamic programming](stochastic-dp.md)
replaces the deterministic continuation value by a conditional expectation.
