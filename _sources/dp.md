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

To ground our discussion, let us return to the domain of discrete-time optimal control problems (DOCPs). These problems frequently arise from the discretization of continuous-time optimal control problems. While the focus here will be on deterministic problems, it is worth noting that these concepts extend naturally to stochastic problems by taking the expectation over the random quantities.

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
:label: backward-recursion

**Input:** Terminal cost function $c_\mathrm{T}(\cdot)$, stage cost functions $c_t(\cdot, \cdot)$, system dynamics $f_t(\cdot, \cdot)$, time horizon $\mathrm{T}$

**Output:** Optimal value functions $J_t^\star(\cdot)$ and optimal control policies $\mu_t^\star(\cdot)$ for $t = 1, \ldots, T$

1. Initialize $J_T^\star(\mathbf{x}) = c_\mathrm{T}(\mathbf{x})$ for all $\mathbf{x}$ in the state space
2. For $t = T-1, T-2, \ldots, 1$:
   1. For each state $\mathbf{x}$ in the state space:
      1. Compute $J_t^\star(\mathbf{x}) = \min_{\mathbf{u}} \left[ c_t(\mathbf{x}, \mathbf{u}) + J_{t+1}^\star(f_t(\mathbf{x}, \mathbf{u})) \right]$
      2. Compute $\mu_t^\star(\mathbf{x}) = \arg\min_{\mathbf{u}} \left[ c_t(\mathbf{x}, \mathbf{u}) + J_{t+1}^\star(f_t(\mathbf{x}, \mathbf{u})) \right]$
   2. End For
3. End For
4. Return $J_t^\star(\cdot)$, $\mu_t^\star(\cdot)$ for $t = 1, \ldots, T$
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
and select any minimizer $\boldsymbol{\mu}_t^\star(\mathbf{x})\in\arg\min(\cdot)$.

**Claim.** For every initial state $\mathbf{x}_1$, the control sequence
$\boldsymbol{\mu}_1^\star(\mathbf{x}_1),\dots,\boldsymbol{\mu}_{T-1}^\star(\mathbf{x}_{T-1})$
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
By assumption, there exists $\boldsymbol{\mu}_t^\star(\mathbf{x})$ attaining the minimum in the definition of $J_t^\star(\mathbf{x})$. If we now **paste** to $\boldsymbol{\mu}_t^\star(\mathbf{x})$ an optimal tail policy from $t+1$ (whose existence we will establish inductively), the resulting sequence attains cost exactly

$$
c_t\big(\mathbf{x},\boldsymbol{\mu}_t^\star(\mathbf{x})\big)
+J_{t+1}^\star\!\Big(\mathbf{f}_t\big(\mathbf{x},\boldsymbol{\mu}_t^\star(\mathbf{x})\big)\Big)
=J_t^\star(\mathbf{x}),
$$
which matches the lower bound $(\ast)$; hence it is optimal from $t$.

**Step 3 (backward induction over time).**  
Base case $t=T$. The statement holds because $J_T^\star(\mathbf{x})=c_\mathrm{T}(\mathbf{x})$ and there is no control to choose.

Inductive step. Assume the tail statement holds for $t+1$: from any state $\mathbf{x}_{t+1}$ there exists an optimal control sequence realizing $J_{t+1}^\star(\mathbf{x}_{t+1})$. Then by Steps 1–2, selecting $\boldsymbol{\mu}_t^\star(\mathbf{x}_t)$ at stage $t$ and concatenating the optimal tail from $t+1$ yields an optimal sequence from $t$ with value $J_t^\star(\mathbf{x}_t)$.

By backward induction, the claim holds for all $t$, in particular for $t=1$ and any initial $\mathbf{x}_1$. Therefore the backward recursion both **certifies** optimality (verification) and **constructs** an optimal policy (synthesis), while recovering the full family $\{J_t^\star\}_{t=1}^T$.
```

```{prf:remark} No “big NLP” required
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
Replace each “min” by “inf” in the definitions and state that for every $\varepsilon>0$ there exist $\varepsilon$-optimal selectors $\boldsymbol{\mu}_t^\varepsilon(\cdot)$ achieving cost within $\varepsilon$ of $J_t^\star(\cdot)$. The same cut-and-paste and induction go through.
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

```{code-cell} ipython3
:tags: [hide-input]
:load: code/harvest_dp.py

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
To evaluate Bellman’s equation at an arbitrary $\mathbf{x}_{k+1}$, we interpolate. 
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
:label: backward-recursion-interpolation

**Input:** 
- Terminal cost $c_\mathrm{T}(\cdot)$  
- Stage costs $c_t(\cdot,\cdot)$  
- Dynamics $f_t(\cdot,\cdot)$  
- Time horizon $T$  
- State grid $\mathcal{X}_\text{grid}$  
- Action set $\mathcal{U}$  
- Interpolation method $\mathcal{I}(\cdot)$ (e.g., linear, cubic spline, RBF, neural network)

**Output:** Value functions $J_t^\star(\cdot)$ and policies $\mu_t^\star(\cdot)$ for all $t$

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
      - Store optimal action: $\mu_t^\star(\mathbf{x}) = \arg\min_{\mathbf{u} \in \mathcal{U}} J_t(\mathbf{x}, \mathbf{u})$
   
   b. **Fit interpolator for current stage:**  
      $I_t \leftarrow \mathcal{I}(\{\mathbf{x}, J_t^\star(\mathbf{x})\}_{\mathbf{x} \in \mathcal{X}_\text{grid}})$

3. **Return:** Interpolated value functions $\{I_t\}_{t=0}^T$ and policies $\{\mu_t^\star\}_{t=0}^{T-1}$
```


#### Example: Optimal Harvest with Linear Interpolation

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

<!-- ## Linear Quadratic Regulator via Dynamic Programming

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
$$ -->
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

$J = \sum_{k=0}^{\infty} \left( x^\mathrm{T}(k)Qx(k) + ru^2(k) \right)$

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
1. **Markovian Deterministic (MD)**: $d_t: \mathcal{S} \rightarrow \mathcal{A}_s$
2. **Markovian Randomized (MR)**: $d_t: \mathcal{S} \rightarrow \mathcal{P}(\mathcal{A}_s)$
3. **History-dependent Deterministic (HD)**: $d_t: H_t \rightarrow \mathcal{A}_s$
4. **History-dependent Randomized (HR)**: $d_t: H_t \rightarrow \mathcal{P}(\mathcal{A}_s)$

where $\mathcal{P}(\mathcal{A}_s)$ denotes the set of probability distributions over $\mathcal{A}_s$.

A **policy** $\pi$ is a sequence of decision rules, one for each decision epoch:

$$\pi = (d_1, d_2, ..., d_{N-1})$$

The set of all policies of class $K$ (where $K \in \{HR, HD, MR, MD\}$) is denoted as $\Pi^K$. These policy classes form a hierarchy:

$$\Pi^{MD} \subset \Pi^{MR} \subset \Pi^{HR}, \quad \Pi^{MD} \subset \Pi^{HD} \subset \Pi^{HR}$$

The largest set $\Pi^{HR}$ contains all possible policies. A fundamental question in MDP theory is: under what conditions can we restrict attention to the much simpler set $\Pi^{MD}$ without loss of optimality?

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

A remarkable simplification occurs when we examine these history-based equations more closely. The Markov property of our system dynamics and rewards means that the optimal return actually depends on the history only through the current state:

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

**Intuition:** The Markov property means that the current state contains all information needed to predict future evolution. The past provides no additional value for decision-making. This powerful result allows us to work with value functions $v_t^*(s)$ indexed only by state and time, dramatically simplifying both theory and computation.

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
:label: stochastic-backward-recursion

**Input:** Terminal cost function $c_\mathrm{T}(\cdot)$, stage cost functions $c_t(\cdot, \cdot, \cdot)$, system dynamics $\mathbf{f}_t(\cdot, \cdot, \cdot)$, time horizon $\mathrm{T}$, disturbance distributions

**Output:** Optimal value functions $J_t^\star(\cdot)$ and optimal control policies $\mu_t^\star(\cdot)$ for $t = 1, \ldots, T$

1. Initialize $J_T^\star(\mathbf{x}) = c_\mathrm{T}(\mathbf{x})$ for all $\mathbf{x}$ in the state space
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

```{code-cell} ipython3
:tags: [hide-input]
:load: code/harvest_sdp.py
```

## Linear Quadratic Regulator via Dynamic Programming

We now examine a special case where the backward recursion admits a remarkable closed-form solution. When the system dynamics are linear and the cost function is quadratic, the optimization at each stage can be solved analytically. Moreover, the value function itself maintains a quadratic structure throughout the recursion, and the optimal policy reduces to a simple linear feedback law. This elegant result eliminates the need for discretization, interpolation, or any function approximation. The infinite-dimensional problem collapses to tracking a finite set of matrices.

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

What we now have to observe is that if the terminal cost is quadratic, then the value function at every earlier stage remains quadratic. This is not immediately obvious, but it follows from the structure of Bellman's equation combined with the linearity of the dynamics.

We claim that the optimal cost-to-go from any stage $t$ takes the form:

$$
J_t^\star(\mathbf{x}_t) = \frac{1}{2}\mathbf{x}_t^\top P_t \mathbf{x}_t
$$

for some positive semidefinite matrix $P_t$. At the terminal time, this is true by definition: $P_T = Q_T$.

Let's verify this structure and derive the recursion for $P_t$ using backward induction. Suppose we've established that $J_{t+1}^\star(\mathbf{x}_{t+1}) = \frac{1}{2}\mathbf{x}_{t+1}^\top P_{t+1} \mathbf{x}_{t+1}$. Bellman's equation at stage $t$ states:

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


Putting everything together, the backward induction procedure under the LQR setting then becomes: 


````{prf:algorithm} Backward Recursion for LQR
:label: lqr-backward-recursion

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

Now we should highlight that the notation used in control theory tends to differ from that found in operations research communities, in which the field of dynamic programming flourished. We summarize those (purely notational) differences in this section.

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

Let's go back to the starting point and define what it means for a policy to be optimal in a Markov Decision Problem. For this, we will be considering different possible search spaces (policy classes) and compare policies based on the ordering of their value from any possible start state. The value of a policy $\pi$ (optimal or not) is defined as the expected total reward obtained by following that policy from a given starting state. Formally, for a finite-horizon MDP with $N$ decision epochs, we define the value function $v_\pi(s, t)$ as:

$$
v_\pi(s, t) \triangleq \mathbb{E}\left[\sum_{k=t}^{N-1} r_t(S_k, A_k) + r_N(S_N) \mid S_t = s\right]
$$

where $S_t$ is the state at time $t$, $A_t$ is the action taken at time $t$, and $r_t$ is the reward function. For simplicity, we write $v_\pi(s)$ to denote $v_\pi(s, 1)$, the value of following policy $\pi$ from state $s$ at the first stage over the entire horizon $N$.

In finite-horizon MDPs, our goal is to identify an optimal policy, denoted by $\pi^*$, that maximizes total expected reward over the horizon $N$. Specifically:

$$
v_{\pi^*}(s) \geq v_\pi(s), \quad \forall s \in \mathcal{S}, \quad \forall \pi \in \Pi^{\text{HR}}
$$

We call $\pi^*$ an **optimal policy** because it yields the highest possible value across all states and all policies within the policy class $\Pi^{\text{HR}}$. We denote by $v^*$ the maximum value achievable by any policy:

$$
v^*(s) = \max_{\pi \in \Pi^{\text{HR}}} v_\pi(s), \quad \forall s \in \mathcal{S}
$$

In reinforcement learning literature, $v^*$ is typically referred to as the "optimal value function," while in some operations research references, it might be called the "value of an MDP." An optimal policy $\pi^*$ is one for which its value function equals the optimal value function:

$$
v_{\pi^*}(s) = v^*(s), \quad \forall s \in \mathcal{S}
$$

It's important to note that this notion of optimality applies to every state. Policies optimal in this sense are sometimes called "uniformly optimal policies." A weaker notion of optimality, often encountered in reinforcement learning practice, is optimality with respect to an initial distribution of states. In this case, we seek a policy $\pi \in \Pi^{\text{HR}}$ that maximizes:

$$
\sum_{s \in \mathcal{S}} v_\pi(s) P_1(S_1 = s)
$$

where $P_1(S_1 = s)$ is the probability of starting in state $s$.

A fundamental result in MDP theory states that the maximum value can be achieved by searching over the space of deterministic Markovian Policies. Consequently:

$$ v^*(s) = \max_{\pi \in \Pi^{\mathrm{HR}}} v_\pi(s) = \max _{\pi \in \Pi^{M D}} v_\pi(s), \quad s \in S$$

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
- A markovian deterministic policy $\pi$

**Output:** Value function $v^\pi$ for policy $\pi$

1. Initialize:
   - Set $t = N$
   - For all $s_N \in S$:

     $$v_\pi(s_N, N) = r_N(s_N)$$

2. For $t = N-1$ to $1$:
   - For each $s_t \in S$:
     a. Compute the value function for the given policy:

        $$v_\pi(s_t, t) = r_t(s_t, d_t(s_t)) + \sum_{j \in S} p_t(j | s_t, d_t(s_t)) v_\pi(j, t+1)$$

3. Return the value function $v^\pi(s_t, t)$ for all $t$ and $s_t$
````

This code could also finally be adapted to support randomized policies using:

$$v_\pi(s_t, t) = \sum_{a_t \in \mathcal{A}_{s_t}} d_t(a_t \mid s_t) \left( r_t(s_t, a_t) + \sum_{j \in S} p_t(j | s_t, a_t) v_\pi(j, t+1) \right)$$


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

   - $r(s_i, n_i) = -c_i(n_i)$ for $i = 1, 2, 3$, where $c_i(n_i)$ is the cost of running a trial with sample size $n_i$.
   - $r(s_4) = g_4$, where $g_4$ is the expected profit from a successful drug.

5. **Discount Factor** ($\gamma$): We use a discount factor $0 < \gamma \leq 1$ to account for the time value of money and risk preferences.

```{code-cell} ipython3
:tags: [hide-input]
:load: code/sample_size_drug_dev_dp.py

```

# Infinite-Horizon MDPs

It often makes sense to model control problems over infinite horizons. We extend the previous setting and define the expected total reward of policy $\pi \in \Pi^{\mathrm{HR}}$, $v^\pi$ as:

$$
v^\pi(s) = \mathbb{E}\left[\sum_{t=1}^{\infty} r(S_t, A_t)\right]
$$

One drawback of this model is that we could easily encounter values that are $+\infty$ or $-\infty$, even in a setting as simple as a single-state MDP which loops back into itself and where the accrued reward is nonzero.

Therefore, it is often more convenient to work with an alternative formulation which guarantees the existence of a limit: the expected total discounted reward of policy $\pi \in \Pi^{\mathrm{HR}}$ is defined to be:

$$
v_\gamma^\pi(s) \equiv \lim_{N \rightarrow \infty} \mathbb{E}\left[\sum_{t=1}^N \gamma^{t-1} r(S_t, A_t)\right]
$$

for $0 \leq \gamma < 1$ and when $\max_{s \in \mathcal{S}} \max_{a \in \mathcal{A}_s}|r(s, a)| = R_{\max} < \infty$, in which case, $|v_\gamma^\pi(s)| \leq (1-\gamma)^{-1} R_{\max}$.


Finally, another possibility for the infinite-horizon setting is the so-called average reward or gain of policy $\pi \in \Pi^{\mathrm{HR}}$ defined as:

$$
g^\pi(s) \equiv \lim_{N \rightarrow \infty} \frac{1}{N} \mathbb{E}\left[\sum_{t=1}^N r(S_t, A_t)\right]
$$

We won't be working with this formulation in this course due to its inherent practical and theoretical complexities. 

Extending the previous notion of optimality from finite-horizon models, a policy $\pi^*$ is said to be discount optimal for a given $\gamma$ if: 

$$
v_\gamma^{\pi^*}(s) \geq v_\gamma^\pi(s) \quad \text { for each } s \in S \text { and all } \pi \in \Pi^{\mathrm{HR}}
$$

Furthermore, the value of a discounted MDP $v_\gamma^*(s)$, is defined by:

$$
v_\gamma^*(s) \equiv \max _{\pi \in \Pi^{\mathrm{HR}}} v_\gamma^\pi(s)
$$

More often, we refer to $v_\gamma$ by simply calling it the optimal value function. 

As for the finite-horizon setting, the infinite horizon discounted model does not require history-dependent policies, since for any $\pi \in \Pi^{HR}$ there exists a $\pi^{\prime} \in \Pi^{MR}$ with identical total discounted reward:
$$
v_\gamma^*(s) \equiv \max_{\pi \in \Pi^{HR}} v_\gamma^\pi(s)=\max_{\pi \in \Pi^{MR}} v_\gamma^\pi(s) .
$$

## Random Horizon Interpretation of Discounting
The use of discounting can be motivated both from a modeling perspective and as a means to ensure that the total reward remains bounded. From the modeling perspective, we can view discounting as a way to weight more or less importance on the immediate rewards vs. the long-term consequences. There is also another interpretation which stems from that of a finite horizon model but with an uncertain end time. More precisely:

Let $v_\nu^\pi(s)$ denote the expected total reward obtained by using policy $\pi$ when the horizon length $\nu$ is random. We define it by:

$$
v_\nu^\pi(s) \equiv \mathbb{E}_s^\pi\left[\mathbb{E}_\nu\left\{\sum_{t=1}^\nu r(S_t, A_t)\right\}\right]
$$


````{prf:theorem} Random horizon interpretation of discounting
:label: prop-5-3-1
Suppose that the horizon $\nu$ follows a geometric distribution with parameter $\gamma$, $0 \leq \gamma < 1$, independent of the policy such that 
$P(\nu=n) = (1-\gamma) \gamma^{n-1}, \, n=1,2, \ldots$, then $v_\nu^\pi(s) = v_\gamma^\pi(s)$ for all $s \in \mathcal{S}$ .
````

````{prf:proof}
See proposition 5.3.1 in {cite}`Puterman1994`.

By definition of the finite-horizon value function and the law of total expectation:

$$
v_\nu^\pi(s) = \sum_{n=1}^{\infty} P(\nu=n) \cdot v_n^\pi(s) = \sum_{n=1}^{\infty} (1-\gamma) \gamma^{n-1} \cdot E_s^\pi \left\{\sum_{t=1}^n r(S_t, A_t)\right\}.
$$

Combining the expectation with the sum over $n$:

$$
v_\nu^\pi(s) = E_s^\pi \left\{\sum_{n=1}^{\infty} (1-\gamma) \gamma^{n-1} \sum_{t=1}^n r(S_t, A_t)\right\}.
$$

**Reordering the summations:** Under the bounded reward assumption $|r(s,a)| \leq R_{\max}$ and $\gamma < 1$, we have

$$
E_s^\pi \left\{\sum_{n=1}^{\infty} \sum_{t=1}^n |r(S_t, A_t)| \cdot (1-\gamma) \gamma^{n-1}\right\} \leq R_{\max} \sum_{n=1}^{\infty} n (1-\gamma) \gamma^{n-1} = \frac{R_{\max}}{1-\gamma} < \infty,
$$
which justifies exchanging the order of summation by Fubini's theorem.

To reverse the order, note that the pair $(n,t)$ with $1 \leq t \leq n$ can be reindexed by fixing $t$ first and letting $n$ range from $t$ to $\infty$:

$$
\sum_{n=1}^{\infty} \sum_{t=1}^n = \sum_{t=1}^{\infty} \sum_{n=t}^{\infty}.
$$

Therefore:
\begin{align*}
v_\nu^\pi(s) &= E_s^\pi \left\{\sum_{t=1}^{\infty} r(S_t, A_t) \sum_{n=t}^{\infty} (1-\gamma) \gamma^{n-1}\right\}.
\end{align*}

**Evaluating the inner sum:** Using the substitution $m = n - t + 1$ (so $n = m + t - 1$):
\begin{align*}
\sum_{n=t}^{\infty} (1-\gamma) \gamma^{n-1} &= \sum_{m=1}^{\infty} (1-\gamma) \gamma^{m+t-2} \\
&= \gamma^{t-1} (1-\gamma) \sum_{m=1}^{\infty} \gamma^{m-1} \\
&= \gamma^{t-1} (1-\gamma) \cdot \frac{1}{1-\gamma} = \gamma^{t-1}.
\end{align*}

Substituting back:

$$
v_\nu^\pi(s) = E_s^\pi \left\{\sum_{t=1}^{\infty} \gamma^{t-1} r(S_t, A_t)\right\} = v_\gamma^\pi(s).
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

For a Markovian decision rule $d \in D^{MD}$, we define:

\begin{align*}
\mathbf{r}_d(s) &\equiv r(s, d(s)), \quad \mathbf{r}_d \in \mathbb{R}^{|S|}, \\
[\mathbf{P}_d]_{s,j} &\equiv p(j \mid s, d(s)), \quad \mathbf{P}_d \in \mathbb{R}^{|S| \times |S|}.
\end{align*}

For a randomized decision rule $d \in D^{MR}$, these definitions extend to:

\begin{align*}
\mathbf{r}_d(s) &\equiv \sum_{a \in A_s} d(a \mid s) \, r(s, a), \\
[\mathbf{P}_d]_{s,j} &\equiv \sum_{a \in A_s} d(a \mid s) \, p(j \mid s, a).
\end{align*}

In both cases, $\mathbf{r}_d$ denotes a reward vector in $\mathbb{R}^{|S|}$, with each component $\mathbf{r}_d(s)$ representing the reward associated with state $s$. Similarly, $\mathbf{P}_d$ is a transition probability matrix in $\mathbb{R}^{|S| \times |S|}$, capturing the transition probabilities under decision rule $d$.

For a nonstationary Markovian policy $\pi = (d_1, d_2, \ldots) \in \Pi^{MR}$, the expected total discounted reward is given by:

$$
\mathbf{v}_\gamma^{\pi}(s)=\mathbb{E}\left[\sum_{t=1}^{\infty} \gamma^{t-1} r\left(S_t, A_t\right) \,\middle|\, S_1 = s\right].
$$

Using vector notation, this can be expressed as:

$$
\begin{aligned}
\mathbf{v}_\gamma^{\pi} &= \sum_{t=1}^{\infty} \gamma^{t-1} \mathbf{P}_\pi^{t-1} \mathbf{r}_{d_1} \\
&= \mathbf{r}_{d_1} + \gamma \mathbf{P}_{d_1} \mathbf{r}_{d_2} + \gamma^2 \mathbf{P}_{d_1} \mathbf{P}_{d_2} \mathbf{r}_{d_3} + \cdots \\
&= \mathbf{r}_{d_1} + \gamma \mathbf{P}_{d_1} \left( \mathbf{r}_{d_2} + \gamma \mathbf{P}_{d_2} \mathbf{r}_{d_3} + \gamma^2 \mathbf{P}_{d_2} \mathbf{P}_{d_3} \mathbf{r}_{d_4} + \cdots \right).
\end{aligned}
$$

This formulation leads to a recursive relationship:

$$
\begin{align*}
\mathbf{v}_\gamma^\pi &= \mathbf{r}_{d_1} + \gamma \mathbf{P}_{d_1} \mathbf{v}_\gamma^{\pi^{\prime}}\\
&=\sum_{t=1}^{\infty} \gamma^{t-1} \mathbf{P}_\pi^{t-1} \mathbf{r}_{d_t}
\end{align*}
$$

where $\pi^{\prime} = (d_2, d_3, \ldots)$.


For stationary policies, where $\pi = d^{\infty} \equiv (d, d, \ldots)$, the total expected reward simplifies to:

$$
\begin{align*}
\mathbf{v}_\gamma^{d^\infty} &= \mathbf{r}_d+ \gamma \mathbf{P}_d \mathbf{v}_\gamma^{d^\infty} \\
&=\sum_{t=1}^{\infty} \gamma^{t-1} \mathbf{P}_d^{t-1} \mathbf{r}_{d}
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

We can now verify that $(\mathbf{I} - \gamma \mathbf{P}_d)$ is invertible and the Neumann series converges.

1. **Norm of the transition matrix:** Since $\mathbf{P}_d$ is a stochastic matrix (each row sums to 1 and all entries are non-negative), its $\ell_\infty$-norm is:

   $$
   \|\mathbf{P}_d\| = \max_{s \in S} \sum_{j \in S} [\mathbf{P}_d]_{s,j} = \max_{s \in S} 1 = 1.
   $$

2. **Norm of the scaled matrix:** Using the homogeneity property of norms, we have:

   $$
   \|\gamma \mathbf{P}_d\| = |\gamma| \cdot \|\mathbf{P}_d\| = |\gamma| \cdot 1 = |\gamma|.
   $$

3. **Bounding the spectral radius:** Applying the fundamental inequality between spectral radius and matrix norm:

   $$
   \rho(\gamma \mathbf{P}_d) \leq \|\gamma \mathbf{P}_d\| = |\gamma|.
   $$

4. **Verifying convergence:** Since $0 \leq \gamma < 1$ by assumption, we have:

   $$
   \rho(\gamma \mathbf{P}_d) \leq |\gamma| < 1.
   $$
   
   This strict inequality guarantees that $(\mathbf{I} - \gamma \mathbf{P}_d)$ is invertible and the Neumann series converges.

Therefore, the Neumann series expansion converges and yields:

$$
\mathbf{v}_\gamma^{d^\infty} = (\mathbf{I} - \gamma \mathbf{P}_d)^{-1} \mathbf{r}_d = \sum_{t=0}^{\infty} (\gamma \mathbf{P}_d)^t \mathbf{r}_d = \sum_{t=1}^{\infty} \gamma^{t-1} \mathbf{P}_d^{t-1} \mathbf{r}_d.
$$

Consequently, for a stationary policy, $\mathbf{v}_\gamma^{d^\infty}$ can be determined as the solution to the linear equation:

$$
\mathbf{v} = \mathbf{r}_d+ \gamma \mathbf{P}_d\mathbf{v},
$$

which can be rearranged to:

$$
(\mathbf{I} - \gamma \mathbf{P}_d) \mathbf{v} = \mathbf{r}_d.
$$

We can also characterize $\mathbf{v}_\gamma^{d^\infty}$ as the solution to an operator equation. More specifically, define the transformation $\mathrm{L}_d$ by

$$
\mathrm{L}_d \mathbf{v} \equiv \mathbf{r}_d+\gamma \mathbf{P}_d\mathbf{v}
$$

for any $\mathbf{v} \in V$. Intuitively, $\mathrm{L}_d$ takes a value function $\mathbf{v}$ as input and returns a new value function that combines immediate rewards ($\mathbf{r}_d$) with discounted future values ($\gamma \mathbf{P}_d\mathbf{v}$). 

```{note}
While we often refer to $\mathrm{L}_d$ as a "linear operator" in the RL literature, it is technically an **affine operator** (or affine transformation), not a linear operator in the strict sense. To see why, recall that a linear operator $\mathcal{T}$ must satisfy:

1. **Additivity:** $\mathcal{T}(\mathbf{v}_1 + \mathbf{v}_2) = \mathcal{T}(\mathbf{v}_1) + \mathcal{T}(\mathbf{v}_2)$
2. **Homogeneity:** $\mathcal{T}(\alpha \mathbf{v}) = \alpha \mathcal{T}(\mathbf{v})$ for all scalars $\alpha$

However, $\mathrm{L}_d$ fails the additivity test:

$$
\mathrm{L}_d(\mathbf{v}_1 + \mathbf{v}_2) = \mathbf{r}_d + \gamma \mathbf{P}_d(\mathbf{v}_1 + \mathbf{v}_2) = \mathbf{r}_d + \gamma \mathbf{P}_d\mathbf{v}_1 + \gamma \mathbf{P}_d\mathbf{v}_2
$$

while

$$
\mathrm{L}_d(\mathbf{v}_1) + \mathrm{L}_d(\mathbf{v}_2) = (\mathbf{r}_d + \gamma \mathbf{P}_d\mathbf{v}_1) + (\mathbf{r}_d + \gamma \mathbf{P}_d\mathbf{v}_2) = 2\mathbf{r}_d + \gamma \mathbf{P}_d\mathbf{v}_1 + \gamma \mathbf{P}_d\mathbf{v}_2.
$$

The presence of the constant term $\mathbf{r}_d$ makes $\mathrm{L}_d$ affine rather than linear. An affine operator has the form $\mathcal{A}(\mathbf{v}) = \mathbf{b} + \mathcal{T}(\mathbf{v})$, where $\mathbf{b}$ is a constant vector and $\mathcal{T}$ is a linear operator. In our case, $\mathbf{b} = \mathbf{r}_d$ and $\mathcal{T}(\mathbf{v}) = \gamma \mathbf{P}_d\mathbf{v}$.

Despite this technical distinction, the term "linear operator" is commonly used in the reinforcement learning literature when referring to $\mathrm{L}_d$, following a slight abuse of terminology.
```

Therefore, we view $\mathrm{L}_d$ as an operator mapping elements of $V$ to $V$: i.e., $\mathrm{L}_d: V \rightarrow V$. The fact that the value function of a policy is the solution to a fixed-point equation can then be expressed with the statement: 

$$
\mathbf{v}_\gamma^{d^\infty}=\mathrm{L}_d \mathbf{v}_\gamma^{d^\infty}.
$$

This is a **fixed-point equation**: the value function $\mathbf{v}_\gamma^{d^\infty}$ is a fixed point of the operator $\mathrm{L}_d$.

## Solving Operator Equations

The operator equation we encountered in MDPs, $\mathbf{v}_\gamma^{d^\infty} = \mathrm{L}_d \mathbf{v}_\gamma^{d^\infty}$, is a specific instance of a more general class of problems known as operator equations. These equations appear in various fields of mathematics and applied sciences, ranging from differential equations to functional analysis.

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

We can adopt an operator-theoretic perspective by defining a nonlinear operator $\mathrm{L}$ on the space $V$ of bounded real-valued functions on the state space $S$. We can then show that the value of an MDP is the solution to the following fixed-point equation:

$$
\mathrm{L} \mathbf{v} \equiv \max_{d \in D^{MD}} \left\{\mathbf{r}_d + \gamma \mathbf{P}_d \mathbf{v}\right\}
$$

where $D^{MD}$ is the set of Markov deterministic decision rules, $\mathbf{r}_d$ is the reward vector under decision rule $d$, and $\mathbf{P}_d$ is the transition probability matrix under decision rule $d$.

Note that while we write $\max_{d \in D^{MD}}$, we do not implement the above operator in this way. Written in this fashion, it would indeed imply that we first need to enumerate all Markov deterministic decision rules and pick the maximum. Now the fact that we compare policies based on their value functions in a componentwise fashion, maxing over the space of Markovian deterministic rules amounts to the following update in component form:

$$
(\mathrm{L} \mathbf{v})(s) = \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right\}
$$

The equivalence between these two forms can be shown mathematically, as demonstrated in the following proposition and proof.

```{prf:proposition}
The operator $\mathrm{L}$ defined as a maximization over Markov deterministic decision rules:

$$(\mathrm{L} \mathbf{v})(s) = \max_{d \in D^{MD}} \left\{r(s,d(s)) + \gamma \sum_{j \in \mathcal{S}} p(j|s,d(s)) v(j)\right\}$$

is equivalent to the componentwise maximization over actions:

$$(\mathrm{L} \mathbf{v})(s) = \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right\}$$
```

```{prf:proof}
Let's define the right-hand side of the first equation as $R_1$ and the right-hand side of the second equation as $R_2$. We'll prove that $R_1 \leq R_2$ and $R_2 \leq R_1$, which will establish their equality.

Step 1: Proving $R_1 \leq R_2$

For any $d \in D^{MD}$, we have:

$$r(s,d(s)) + \gamma \sum_{j \in \mathcal{S}} p(j|s,d(s)) v(j) \leq \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right\}$$

This is because $d(s) \in \mathcal{A}_s$, so the left-hand side is included in the set over which we're maximizing on the right-hand side.

Taking the maximum over all $d \in D^{MD}$ on the left-hand side doesn't change this inequality:

$$\max_{d \in D^{MD}} \left\{r(s,d(s)) + \gamma \sum_{j \in \mathcal{S}} p(j|s,d(s)) v(j)\right\} \leq \max_{a \in \mathcal{A}_s} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j)\right\}$$

Therefore, $R_1 \leq R_2$.

Step 2: Proving $R_2 \leq R_1$

Let $a^* \in \mathcal{A}_s$ be the action that achieves the maximum in $R_2$. We can construct a Markov deterministic decision rule $d^*$ such that $d^*(s) = a^*$ and $d^*(s')$ is arbitrary for $s' \neq s$. Then:

$$\begin{align*}
R_2 &= r(s,a^*) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a^*) v(j) \\
&= r(s,d^*(s)) + \gamma \sum_{j \in \mathcal{S}} p(j|s,d^*(s)) v(j) \\
&\leq \max_{d \in D^{MD}} \left\{r(s,d(s)) + \gamma \sum_{j \in \mathcal{S}} p(j|s,d(s)) v(j)\right\} \\
&= R_1
\end{align*}$$

Conclusion:
Since we've shown $R_1 \leq R_2$ and $R_2 \leq R_1$, we can conclude that $R_1 = R_2$, which proves the equivalence of the two forms of the operator.
```

## Algorithms for Solving the Optimality Equations

The optimality equations are operator equations. Therefore, we can apply general numerical methods to solve them. Applying the successive approximation method to the Bellman optimality equation yields a method known as "value iteration" in dynamic programming. A direct application of the blueprint for successive approximation yields the following algorithm:

````{prf:algorithm} Value Iteration
:label: value-iteration

**Input** Given an MDP $(S, A, P, R, \gamma)$ and tolerance $\varepsilon > 0$  

**Output** Compute an $\varepsilon$-optimal value function $V$ and policy $\pi$  

1. Initialize $v_0(s) = 0$ for all $s \in S$  
2. $n \leftarrow 0$  
3. **repeat**  

    1. For each $s \in S$:  

        1. $v_{n+1}(s) \leftarrow \max_{a \in A} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)v_n(s')\right\}$  

    2. $\delta \leftarrow \|v_{n+1} - v_n\|_\infty$  
    3. $n \leftarrow n + 1$  

4. **until** $\delta < \frac{\varepsilon(1-\gamma)}{2\gamma}$  
5. For each $s \in S$:  

    1. $\pi(s) \leftarrow \arg\max_{a \in A} \left\{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a)v_n(s')\right\}$  

6. **return** $v_n, \pi$  
````

The termination criterion in this algorithm is based on a specific bound that provides guarantees on the quality of the solution. This is in contrast to supervised learning, where we often use arbitrary termination criteria based on computational budget or early stopping when the learning curve flattens. This is because establishing implementable generalization bounds in supervised learning is challenging.

However, in the dynamic programming context, we can derive various bounds that can be implemented in practice. These bounds help us terminate our procedure with a guarantee on the precision of our value function and, correspondingly, on the optimality of the resulting policy.

````{prf:proposition} Convergence of Value Iteration 
:label: value-iteration-convergence
(Adapted from {cite:t}`Puterman1994` theorem 6.3.1)

Let $v_0$ be any initial value function, $\varepsilon > 0$ a desired accuracy, and let $\{v_n\}$ be the sequence of value functions generated by value iteration, i.e., $v_{n+1} = \mathrm{L}v_n$ for $n \geq 0$, where $\mathrm{L}$ is the Bellman optimality operator. Then:

1. $v_n$ converges to the optimal value function $v^*_\gamma$,
2. The algorithm terminates in finite time,
3. The resulting policy $\pi_\varepsilon$ is $\varepsilon$-optimal, and
4. When the algorithm terminates, $v_{n+1}$ is within $\varepsilon/2$ of $v^*_\gamma$.

````

````{prf:proof}
Parts 1 and 2 follow directly from the fact that $\mathrm{L}$ is a contraction mapping. Hence, by Banach's fixed-point theorem, it has a unique fixed point (which is $v^*_\gamma$), and repeated application of $\mathrm{L}$ will converge to this fixed point. Moreover, this convergence happens at a geometric rate, which ensures that we reach the termination condition in finite time.

To show that the Bellman optimality operator $\mathrm{L}$ is a contraction mapping, we need to prove that for any two value functions $v$ and $u$:

$$\|\mathrm{L}v - \mathrm{L}u\|_\infty \leq \gamma \|v - u\|_\infty$$

where $\gamma \in [0,1)$ is the discount factor and $\|\cdot\|_\infty$ is the supremum norm.

Let's start by writing out the definition of $\mathrm{L}v$ and $\mathrm{L}u$:

$$\begin{align*}
(\mathrm{L}v)(s) &= \max_{a \in A} \left\{r(s,a) + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a)v(s')\right\}\\
(\mathrm{L}u)(s) &= \max_{a \in A} \left\{r(s,a) + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a)u(s')\right\}
\end{align*}$$

For any state $s$, let $a_v$ be the action that achieves the maximum for $(\mathrm{L}v)(s)$, and $a_u$ be the action that achieves the maximum for $(\mathrm{L}u)(s)$. By the definition of these maximizers:

$$\begin{align*}
(\mathrm{L}v)(s) &\geq r(s,a_u) + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a_u)v(s')\\
(\mathrm{L}u)(s) &\geq r(s,a_v) + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a_v)u(s')
\end{align*}$$

Subtracting these inequalities:

$$\begin{align*}
(\mathrm{L}v)(s) - (\mathrm{L}u)(s) &\leq \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a_v)(v(s') - u(s'))\\
(\mathrm{L}u)(s) - (\mathrm{L}v)(s) &\leq \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a_u)(u(s') - v(s'))
\end{align*}$$

Taking the absolute value and using the fact that $\sum_{s' \in \mathcal{S}} p(s'|s,a) = 1$:

$$|(\mathrm{L}v)(s) - (\mathrm{L}u)(s)| \leq \gamma \max_{s' \in \mathcal{S}} |v(s') - u(s')| = \gamma \|v - u\|_\infty$$

Since this holds for all $s \in \mathcal{S}$, taking the supremum over $s$ gives:

$$\|\mathrm{L}v - \mathrm{L}u\|_\infty \leq \gamma \|v - u\|_\infty$$

Thus, $\mathrm{L}$ is a contraction mapping with contraction factor $\gamma$.

Now, let's prove parts 3 and 4. Suppose the algorithm has just terminated, i.e., $\|v_{n+1} - v_n\|_\infty < \frac{\varepsilon(1-\gamma)}{2\gamma}$ for some $n$. We want to show that our current value function $v_{n+1}$ and the policy $\pi_\varepsilon$ derived from it are close to optimal.

By the triangle inequality:

$$\|v^{\pi_\varepsilon}_\gamma - v^*_\gamma\|_\infty \leq \|v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty + \|v_{n+1} - v^*_\gamma\|_\infty$$

For the first term, since $v^{\pi_\varepsilon}_\gamma$ is the fixed point of $\mathrm{L}_{\pi_\varepsilon}$ and $\pi_\varepsilon$ is greedy with respect to $v_{n+1}$ (i.e., $\mathrm{L}_{\pi_\varepsilon}v_{n+1} = \mathrm{L}v_{n+1}$):

$$
\begin{aligned}
\|v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty &= \|\mathrm{L}_{\pi_\varepsilon}v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty \\
&\leq \|\mathrm{L}_{\pi_\varepsilon}v^{\pi_\varepsilon}_\gamma - \mathrm{L}_{\pi_\varepsilon}v_{n+1}\|_\infty + \|\mathrm{L}_{\pi_\varepsilon}v_{n+1} - v_{n+1}\|_\infty \\
&= \|\mathrm{L}_{\pi_\varepsilon}v^{\pi_\varepsilon}_\gamma - \mathrm{L}_{\pi_\varepsilon}v_{n+1}\|_\infty + \|\mathrm{L}v_{n+1} - v_{n+1}\|_\infty \\
&\leq \gamma\|v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty + \gamma\|v_{n+1} - v_n\|_\infty
\end{aligned}
$$

where we used that both $\mathrm{L}$ and $\mathrm{L}_{\pi_\varepsilon}$ are contractions with factor $\gamma$, and that $v_{n+1} = \mathrm{L}v_n$.

Rearranging:

$$\|v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty \leq \frac{\gamma}{1-\gamma}\|v_{n+1} - v_n\|_\infty$$

Similarly, since $v^*_\gamma$ is the fixed point of $\mathrm{L}$:

$$\|v_{n+1} - v^*_\gamma\|_\infty = \|\mathrm{L}v_n - \mathrm{L}v^*_\gamma\|_\infty \leq \gamma\|v_n - v^*_\gamma\|_\infty \leq \frac{\gamma}{1-\gamma}\|v_{n+1} - v_n\|_\infty$$

Since $\|v_{n+1} - v_n\|_\infty < \frac{\varepsilon(1-\gamma)}{2\gamma}$:

$$\|v^{\pi_\varepsilon}_\gamma - v_{n+1}\|_\infty \leq \frac{\gamma}{1-\gamma} \cdot \frac{\varepsilon(1-\gamma)}{2\gamma} = \frac{\varepsilon}{2}$$

$$\|v_{n+1} - v^*_\gamma\|_\infty \leq \frac{\gamma}{1-\gamma} \cdot \frac{\varepsilon(1-\gamma)}{2\gamma} = \frac{\varepsilon}{2}$$

Combining these:

$$\|v^{\pi_\varepsilon}_\gamma - v^*_\gamma\|_\infty \leq \frac{\varepsilon}{2} + \frac{\varepsilon}{2} = \varepsilon$$

This completes the proof, showing that $v_{n+1}$ is within $\varepsilon/2$ of $v^*_\gamma$ (part 4) and $\pi_\varepsilon$ is $\varepsilon$-optimal (part 3).
````

### Newton-Kantorovich Applied to Bellman Optimality

We now apply the Newton-Kantorovich framework to the Bellman optimality equation. Let

$$
(\mathrm{L}v)(s) = \max_{a \in A(s)} \left\{ r(s,a) + \gamma \sum_{s'} p(s' \mid s,a) v(s') \right\}.
$$

The problem is to find $v$ such that $\mathrm{L}v = v$, or equivalently $\mathrm{B}(v) := \mathrm{L}v - v = 0$. The operator $\mathrm{L}$ is piecewise affine, hence not globally differentiable, but it is directionally differentiable everywhere in the Hadamard sense and Fréchet differentiable at points where the maximizer is unique.

We consider three complementary perspectives for understanding and computing its derivative.

#### Perspective 1: Max of Affine Maps

In tabular form, for finite state and action spaces, the Bellman operator can be written as a pointwise maximum of affine maps:

$$
(\mathrm{L}v)(s) = \max_{a \in A(s)} \left\{ r(s,a) + \gamma (P_a v)(s) \right\},
$$
where $P_a \in \mathbb{R}^{|S| \times |S|}$ is the transition matrix associated with action $a$. Each $Q_a v := r^a + \gamma P_a v$ is affine in $v$. The operator $\mathrm{L}$ therefore computes the upper envelope of a finite set of affine functions at each state.

At any $v$, let the **active set** at state $s$ be

$$
\mathcal{A}^*(s; v) := \arg\max_{a \in A(s)} (Q_a v)(s).
$$

Then the Hadamard directional derivative exists and is given by

$$
(\mathrm{L}'(v; h))(s) = \max_{a \in \mathcal{A}^*(s; v)} \gamma (P_a h)(s).
$$

If the active set is a singleton, this expression becomes linear in $h$, and $\mathrm{L}$ is Fréchet differentiable at $v$, with

$$
\mathrm{L}'(v) = \gamma P_{\pi_v},
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
v_i(c) := (\mathrm{L}v_c)(s_i) = \max_{a \in A(s_i)} \left\{ r(s_i,a) + \gamma \sum_j c_j \mathbb{E}_{s' \mid s_i, a}[\phi_j(s')] \right\}.
$$

Define

$$
F_i(a, c) := r(s_i,a) + \gamma \sum_j c_j \mathbb{E}_{s' \mid s_i, a}[\phi_j(s')],
$$

so that $v_i(c) = \max_a F_i(a, c)$. Since $F_i$ is linear in $c$, we can apply the **envelope theorem** (Danskin's theorem): if the optimizer $a_i^*(c)$ is unique or selected measurably, then

$$
\frac{\partial v_i}{\partial c_j}(c) = \gamma \mathbb{E}_{s' \mid s_i, a_i^*(c)}[\phi_j(s')].
$$

The key is that we do not need to differentiate the optimizer $a_i^*(c)$ itself. The result extends to the subdifferential case when ties occur, where the Jacobian becomes set-valued.

This result is useful when solving the collocation equation $\Phi c = v(c)$. Newton's method requires the Jacobian $v'(c)$, and this expression allows us to compute it without involving any derivatives of the optimal action.

#### Perspective 3: The Implicit Function Theorem

The third perspective applies the implicit function theorem to understand when the Bellman operator is differentiable despite containing a max operator. The key insight is that the maximization problem defines an implicit relationship between the value function and the optimal action, and the implicit function theorem tells us when this relationship is smooth enough to differentiate through.

The Bellman operator is defined as

$$
(\mathrm{L}v)(s) = \max_{a} \left\{ r(s,a) + \gamma \sum_j p(j \mid s,a) v(j) \right\}.
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

This strict inequality is the regularity condition needed for the implicit function theorem. It ensures that the optimal action is not only unique at $v$, but remains so in a neighborhood of $v$.

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
(\mathrm{L}v)(s) = Q_{a^*(s)}(v, s) = r(s,a^*(s)) + \gamma \sum_j p(j \mid s,a^*(s)) v(j)
$$
as an explicit formula that holds throughout the neighborhood. Since $Q_{a^*(s)}(\cdot, s)$ is an affine (hence smooth) function of $v$, we can differentiate it:

$$
\frac{d}{dv} (\mathrm{L}v)(s) = \gamma P_{a^*(s)}.
$$

More precisely, for any perturbation $h$:

$$
(\mathrm{L}(v+h))(s) = (\mathrm{L}v)(s) + \gamma \sum_j p(j \mid s,a^*(s)) h(j) + o(\|h\|).
$$

This is the Fréchet derivative:

$$
\mathrm{L}'(v) = \gamma P_{\pi_v},
$$

where $\pi_v(s) = a^*(s)$ is the greedy policy.

The implicit function theorem guarantees that when the maximizer is unique with a strict gap (the regularity condition), the argmax function $v \mapsto a^*(s; v)$ is locally constant, which removes the non-differentiability of the max operator. Without this regularity condition (specifically, at points where multiple actions tie for optimality), the implicit function theorem does not apply, and the operator is not Fréchet differentiable. The active set perspective (Perspective 1) and the envelope theorem (Perspective 2) provide the tools to handle these non-smooth points.

### Connection to Policy Iteration

We return to the Newton-Kantorovich step:

$$
(I - \mathrm{L}'(v_n)) h_n = v_n - \mathrm{L}v_n,
\quad
v_{n+1} = v_n - h_n.
$$

Suppose $\mathrm{L}'(v_n) = \gamma P_{\pi_{v_n}}$ for the greedy policy $\pi_{v_n}$. Then

$$
(I - \gamma P_{\pi_{v_n}}) v_{n+1} = r^{\pi_{v_n}},
$$

which is exactly policy evaluation for $\pi_{v_n}$. Recomputing the greedy policy from $v_{n+1}$ yields the next iterate.

Thus, **policy iteration is Newton-Kantorovich** applied to the Bellman optimality equation. At points of nondifferentiability (when ties occur), the operator is still semismooth, and policy iteration corresponds to a semismooth Newton method. The envelope theorem is what justifies the simplification of the Jacobian to $\gamma P_{\pi_v}$, bypassing the need to differentiate through the optimizer. This completes the equivalence.

### The Semismooth Newton Perspective

The three perspectives we developed above (the active set view, the envelope theorem, and the implicit function theorem) all point toward a deeper framework for understanding Newton-type methods on non-smooth operators. This framework, known as semismooth Newton methods, was developed precisely to handle operators like the Bellman operator that are piecewise smooth but not globally differentiable. The connection between policy iteration and semismooth Newton methods has been rigorously developed in recent work {cite}`Gargiani2022`.

The classical Newton-Kantorovich method assumes the operator is Fréchet differentiable everywhere. The derivative exists, is unique, and varies continuously with the base point. But the Bellman operator $\mathrm{L}$ violates this assumption at any value function where multiple actions tie for optimality at some state. At such points, the implicit function theorem fails, and there is no unique Fréchet derivative. 

Semismooth Newton methods address this by replacing the notion of a single Jacobian with a generalized derivative that captures the behavior of the operator near non-smooth points. The most commonly used generalized derivative is the Clarke subdifferential, which we can think of as the convex hull of all possible "candidate Jacobians" that arise from limits approaching the non-smooth point from different directions.

For the Bellman residual $\mathrm{B}(v) = \mathrm{L}v - v$, the Clarke subdifferential at a point $v$ can be characterized explicitly using our first perspective. Recall that at each state $s$, we defined the active set $\mathcal{A}^*(s; v) = \arg\max_a Q_a(v, s)$. When this set contains multiple actions, the operator is not Fréchet differentiable. However, it remains directionally differentiable in all directions, and the Clarke subdifferential consists of all matrices of the form

$$
\partial \mathrm{B}(v) = \left\{ I - \gamma P_\pi : \pi(s) \in \mathcal{A}^*(s; v) \text{ for all } s \right\}.
$$

In words, the generalized Jacobian is the set of all matrices $I - \gamma P_\pi$ where $\pi$ is any policy that selects an action from the active set at each state. When the maximizer is unique everywhere, this set reduces to a singleton, and we recover the classical Fréchet derivative. When ties occur, the set has multiple elements: precisely the convex combinations mentioned in Perspective 1.

The semismooth Newton method for solving $\mathrm{B}(v) = 0$ proceeds by selecting an element $J_k \in \partial \mathrm{B}(v_k)$ at each iteration and solving

$$
J_k h_k = -\mathrm{B}(v_k), \quad v_{k+1} = v_k + h_k.
$$

The main takeaway is that any choice from the Clarke subdifferential yields a valid Newton-like update. In the context of the Bellman equation, choosing $J_k = I - \gamma P_{\pi_k}$ where $\pi_k$ is any greedy policy corresponds exactly to the policy evaluation step in policy iteration. The freedom in selecting which action to choose when ties occur translates to the freedom in selecting which element of the subdifferential to use.

Under appropriate regularity conditions (specifically, when the residual function is BD-regular or CD-regular), the semismooth Newton method converges locally at a quadratic rate {cite}`Gargiani2022`. This means that near the solution, the error decreases quadratically:
$$
\|v_{k+1} - v^*\| \leq C \|v_k - v^*\|^2.
$$

This theoretical result explains an empirical observation that has long been noted in practice: policy iteration typically converges in remarkably few iterations, often just a handful, even when the state and action spaces are enormous and the space of possible policies is exponentially large. 

The semismooth Newton framework also suggests a spectrum of methods interpolating between value iteration and policy iteration. Value iteration can be interpreted as a Newton-like method where we choose $J_k = I$ at every iteration, ignoring the dependence of $\mathrm{L}$ on $v$ entirely. This choice guarantees global convergence through the contraction property but sacrifices the quadratic local convergence rate. Policy iteration, at the other extreme, uses the full generalized Jacobian $J_k = I - \gamma P_{\pi_k}$, achieving quadratic convergence but at the cost of solving a linear system at each iteration.

Between these extremes lie methods that use approximate Jacobians. One natural variant is to choose $J_k = \alpha I$ for some scalar $\alpha > 1$. This leads to the update

$$
v_{k+1} = \frac{\alpha - 1}{\alpha} v_k + \frac{1}{\alpha} \mathrm{L}v_k.
$$

This is known as $\alpha$-value iteration or successive over-relaxation when $\alpha > 1$. For appropriate choices of $\alpha$, this method retains global convergence while achieving better local rates than standard value iteration, and it requires only pointwise operations rather than solving a linear system. The Newton perspective thus not only unifies existing algorithms but also generates new ones by systematically exploring different approximations to the generalized Jacobian.

The connection to semismooth Newton methods places policy iteration within a broader mathematical framework that extends far beyond dynamic programming. Semismooth Newton methods are used in optimization (for complementarity problems and variational inequalities), in PDE-constrained optimization (for problems with control constraints), and in economics (for equilibrium problems). The Bellman equation, viewed through this lens, is simply one instance of a piecewise smooth equation, and the tools developed for such equations apply directly.

### Policy Iteration 

While we derived policy iteration-like steps from the Newton-Kantorovich method, it's worth examining policy iteration as a standalone algorithm, as it has been traditionally presented in the field of dynamic programming.

The policy iteration algorithm for discounted Markov decision problems is as follows:

````{prf:algorithm} Policy Iteration
:label: policy-iteration

**Input:** MDP $(S, A, P, R, \gamma)$
**Output:** Optimal policy $\pi^*$

1. Initialize: $n = 0$, select an arbitrary decision rule $d_0 \in D$
2. **repeat**
   3. (Policy evaluation) Obtain $\mathbf{v}^n$ by solving:
   
      $$(\mathbf{I}-\gamma \mathbf{P}_{d_n}) \mathbf{v} = \mathbf{r}_{d_n}$$

   4. (Policy improvement) Choose $d_{n+1}$ to satisfy:

       $$d_{n+1} \in \arg\max_{d \in D}\left\{\mathbf{r}_d+\gamma \mathbf{P}_d \mathbf{v}^n\right\}$$
       
       Set $d_{n+1} = d_n$ if possible.

   5. If $d_{n+1} = d_n$, **return** $d^* = d_n$

   6. $n = n + 1$
7. **until** convergence
````

As opposed to value iteration, this algorithm produces a sequence of both deterministic Markovian decision rules $\{d_n\}$ and value functions $\{\mathbf{v}^n\}$. We recognize in this algorithm the linearization step of the Newton-Kantorovich procedure, which takes place here in the policy evaluation step 3 where we solve the linear system $(\mathbf{I}-\gamma \mathbf{P}_{d_n}) \mathbf{v} = \mathbf{r}_{d_n}$. In practice, this linear sytem could be solved either using direct methods (eg. Gaussian elimination), using  simple iterative methods such as the successive approximation method for policy evaluation, or more sophisticated methods such as GMRES. 

