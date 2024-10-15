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

Unlike the methods we've discussed so far, dynamic programming takes a step back and considers not just a single optimization problem, but an entire family of related problems. This approach, while seemingly more complex at first glance, can often lead to efficient solutions.

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

In other words, any sub-path of an optimal path, from any intermediate point to the end, must itself be optimal.

Dynamic programming can handle nonlinear systems and non-quadratic cost functions naturally. It provides a global optimal solution, when one exists, and can incorporate state and control constraints with relative ease. Howver, as the dimension of the state space increases, this approach suffers from what Bellman termed the "curse of dimensionality." The computational complexity and memory requirements grow exponentially with the state dimension, rendering direct application of dynamic programming intractable for high-dimensional problems.

Fortunately, learning-based methods offer efficient tools to combat the curse of dimensionality on two fronts: by using function approximation (e.g., neural networks) to avoid explicit discretization, and by leveraging randomization through Monte Carlo methods inherent in the learning paradigm. Most of this course is dedicated to those ideas.

## Backward Recursion 

The principle of optimality provides a methodology for solving optimal control problems. Beginning at the final time horizon and working backwards, at each stage the local optimization problem given by Bellman's equation is solved. This process, termed backward recursion or backward induction constructs the optimal value function stage by stage.

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
- Terminal cost function $c_\mathrm{T}(\cdot)$
- Stage cost functions $c_t(\cdot, \cdot)$
- System dynamics $f_t(\cdot, \cdot)$
- Time horizon $\mathrm{T}$
- Grid of state points $\mathcal{X}_\text{grid}$
- Set of possible actions $\mathcal{U}$

**Output:** Optimal value functions $J_t^\star(\cdot)$ and optimal control policies $\mu_t^\star(\cdot)$ for $t = 1, \ldots, T$ at grid points

1. Initialize $J_T^\star(\mathbf{x}) = c_\mathrm{T}(\mathbf{x})$ for all $\mathbf{x} \in \mathcal{X}_\text{grid}$
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

The choice of interpolation method can significantly affect the accuracy of the solution. Linear interpolation is simple and often effective, but higher-order methods like cubic spline interpolation might provide better results in some problems. Furthermore, the layout and density of the grid points in $\mathcal{X}_\text{grid}$ can impact both the accuracy and computational efficiency. A finer grid generally provides better accuracy but increases computational cost. To balance this tradeoff, you might consider techniques like adaptive grid refinement or function approximation methods instead of fixed grid-based interpolation. Special care may also be needed for states near the boundaries, where interpolation might not be possible in all directions.

While simple to implement, interpolation methods scale poorly in multi-dimensional spaces in terms of computational complexity. Techniques like multilinear interpolation with tensorized representations or more advanced methods like radial basis function interpolation might be necessary.

To better address this computational challenge, we will broaden our perspective through the lens of numerical approximation methods for solving functional operator equations. Polynomial interpolation is a form of approximation, with properties akin to generalization in machine learning. By building these connections, we will develop techniques capable of more robustly handling the curse of dimensionality by leveraging the generalization properties of machine learning models, and the "blessing of randomness" inherent in supervised learning and Monte Carlo methods.

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

# Stochastic Dynamic Programming in Control Theory

While our previous discussion centered on deterministic systems, many real-world problems involve uncertainty. Stochastic Dynamic Programming (SDP) extends our framework to handle stochasticity in both the objective function and system dynamics.

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

The backward recursion algorithm for SDP follows a similar structure to its deterministic counterpart, with the key difference being that we now have to compute expected values: 

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

It's important to note that decision rules are stage-wise objects. However, to solve an MDP, we need a strategy for the entire horizon. This is where we make a distinction and introduce the concept of a policy. A policy $\pi$ is a sequence of decision rules, one for each decision epoch:

$$\pi = (d_1, d_2, ..., d_{N-1})$$

Where $N$ is the horizon length (possibly infinite). The set of all policies of class $K$ (where $K$ can be HR, HD, MR, or MD) is denoted as $\Pi^K$.

A special type of policy is a stationary policy, where the same decision rule is used at all epochs: $\pi = (d, d, ...)$, often denoted as $d^\infty$. 

The relationships between these policy classes form a hierarchy:

$$\begin{align*}
\Pi^{SD} \subset \Pi^{SR} \subset \Pi^{MR} \subset \Pi^{HR}\\
\Pi^{SD} \subset \Pi^{MD} \subset \Pi^{MR} \subset \Pi^{HR} \\
\Pi^{SD} \subset \Pi^{MD} \subset \Pi^{HD} \subset \Pi^{HR}
\end{align*}
$$

Where SD stands for Stationary Deterministic and SR for Stationary Randomized. The largest set is by far the set of history randomized policies. 

A fundamental question in MDP theory is: under what conditions can we avoid working with the set $\Pi^{HR}$ and focus for example on the much simpler set of deterministic Markovian policy? Even more so, we will see that in the infinite horizon case, we can drop the dependance on time and simply consider stationary deterministic Markovian policies. 
Certainly, I can help clean up and refine this draft. Here's an improved version:

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
See proposition 5.3.1 in {cite}`Puterman1994`

$$
v_\nu^\pi(s) = E_s^\pi \left\{\sum_{n=1}^{\infty} \sum_{t=1}^n r(X_t, Y_t)(1-\gamma) \gamma^{n-1}\right\}.
$$

Under the bounded reward assumption and $\gamma < 1$, the series converges and we can reverse the order of summation :

\begin{align*}
v_\nu^\pi(s) &= E_s^\pi \left\{\sum_{t=1}^{\infty} \sum_{n=t}^{\infty} r(S_t, A_t)(1-\gamma) \gamma^{n-1}\right\} \\
&= E_s^\pi \left\{\sum_{t=1}^{\infty} \gamma^{t-1} r(S_t, A_t)\right\} = v_\gamma^\pi(s)
\end{align*}

where the last line follows from the geometric series: 

$$
\sum_{n=1}^{\infty} \gamma^{n-1} = \frac{1}{1-\gamma}
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

This last expression is called a Neumann series expansion, and it's guaranteed to exists under the assumptions of bounded reward and discount factor strictly less than one. Consequently, for a stationary policy, $\mathbf{v}_\gamma^{d^\infty}$ can be determined as the solution to the linear equation:

$$
\mathbf{v} = \mathbf{r}_d+ \gamma \mathbf{P}_d\mathbf{v},
$$

which can be rearranged to:

$$
(\mathbf{I} - \gamma \mathbf{P}_d) \mathbf{v} = \mathbf{r}_d.
$$

We can also characterize $\mathbf{v}_\gamma^{d^\infty}$ as the solution to an operator equation. More specifically, define the linear transformation $\mathrm{L}_d$ by

$$
\mathrm{L}_d \mathbf{v} \equiv \mathbf{r}_d+\gamma \mathbf{P}_dv
$$

for any $v \in V$. Intuitively, $\mathrm{L}_d$ takes a value function $\mathbf{v}$ as input and returns a new value function that combines immediate rewards ($\mathbf{r}_d$) with discounted future values ($\gamma \mathbf{P}_d\mathbf{v}$). 
<!-- 

A key property of $\mathrm{L}_d$ is that it maps bounded functions to bounded functions, provided certain conditions are met. This is formalized in the following lemma:


```{prf:lemma} Bounded Reward and Value Function
:label: bounded-reward-value

Let $S$ be a discrete state space, and assume that the reward function is bounded such that $|r(s, a)| \leq M$ for all actions $a \in A_s$ and states $s \in S$. For any discount factor $\gamma$ where $0 \leq \gamma \leq 1$, and for all value functions $v \in V$ and decision rules $d \in D^{MR}$, the following holds:

$$
\mathbf{r}_d+ \gamma \mathbf{P}_d \mathbf{v} \in V
$$

where $V$ is the space of bounded real-valued functions on $S$, $\mathbf{r}_d$ is the reward vector, and $\mathbf{P}_d$ is the transition probability matrix under decision rule $d$.
```

```{prf:proof}
We will prove this lemma in three steps:

1. First, we'll show that $\mathbf{r}_d\in V$.
2. Then, we'll prove that $\mathbf{P}_dv \in V$ for all $v \in V$.
3. Finally, we'll combine these results to show that $\mathbf{r}_d+ \gamma \mathbf{P}_dv \in V$.

**Step 1: Showing $\mathbf{r}_d\in V$**

Given that $|r(s, a)| \leq M$ for all $a \in A_s$ and $s \in S$, we can conclude that $\|\mathbf{r}_d\| \leq M$ for all $d \in D^{MR}$. This is because $\mathbf{r}_d$ is a vector whose components are weighted averages of $r(s, a)$ values, which are all bounded by $M$. Therefore, $\mathbf{r}_d$ is a bounded function on $S$, meaning $\mathbf{r}_d\in V$.

**Step 2: Showing $\mathbf{P}_d \mathbf{v} \in V$ for all $\mathbf{v} \in V$**

$P_d$ is a probability matrix, which means that each row sums to 1. This property implies that $\|P_d\| = 1$. Now, for any $v \in V$, we have:

$$
\|\mathbf{P}_d\mathbf{v}\| \leq \|P_d\| \|\mathbf{v}\| = \|\mathbf{v}\|
$$

This inequality shows that if $v$ is bounded (which it is, since $v \in V$), then $\mathbf{P}_dv$ is also bounded by the same value. Therefore, $\mathbf{P}_d\mathbf{v} \in V$ for all $\mathbf{v} \in V$.

**Step 3: Combining the results**

We've shown that $\mathbf{r}_d\in V$ and $\mathbf{P}_d \mathbf{v} \in V$. Since $V$ is a vector space, it is closed under addition and scalar multiplication. Therefore, for any $\gamma$ where $0 \leq \gamma \leq 1$, we can conclude that:

$$
\mathbf{r}_d+ \gamma \mathbf{P}_d \mathbf{v} \in V
$$

This completes the proof.
``` -->
Therefore, we view $\mathrm{L}_d$ as an operator mapping elements of $V$ to $V$: ie. $\mathrm{L}_d: V \rightarrow V$. The fact that the value function of a policy is the solution to a system of equations can then be expressed with the statement: 

$$
\mathbf{v}_\gamma^{d^x}=\mathrm{L}_d \mathbf{v}_\gamma^{d^x} \text {. }
$$

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

The Newton-Kantorovich method is a generalization of Newton's method from finite dimensional vector spaces to infinite dimensional function spaces: rather than iterating in the space of vectors, we are iterating in the space of functions. Just as in the finite-dimensional counterpart, the idea is to improve the rate of convergence of our method by taking an "educated guess" on where to move next using a linearization of our operator at the current point. Now the concept of linearization, which is synonymous with derivative, will also require a generalization. Here we are in essence trying to quantify how the output of the operator $\mathrm{T}$ -- a function -- varies as we perturb its input -- also a function. The right generalization here is that of the Fréchet derivative.

Before we delve into the Fréchet derivative, it's important to understand the context in which it operates: Banach spaces. A Banach space is a complete normed vector space: ie a vector space, that has a norm, and which every Cauchy sequence convergeces.   Banach spaces provide a natural generalization of finite-dimensional vector spaces to infinite-dimensional settings. 
The norm in a Banach space allows us to quantify the "size" of functions and the "distance" between functions. This allows us to define notions of continuity, differentiability, and for analyzing the convergence of our method.
Furthermore, the completeness property of Banach spaces ensures that we have a well-defined notion of convergence. 

In the context of the Newton-Kantorovich method, we typically work with an operator $\mathrm{T}: X \to Y$, where both $X$ and $Y$ are Banach spaces and whose Fréchet derivative at a point $x \in X$, denoted $\mathrm{T}'(x)$, is a bounded linear operator from $X$ to $Y$ such that:

$$
\lim_{h \to 0} \frac{\|\mathrm{T}(x + h) - \mathrm{T}(x) - \mathrm{T}'(x)h\|_Y}{\|h\|_X} = 0
$$

where $\|\cdot\|_X$ and $\|\cdot\|_Y$ are the norms in $X$ and $Y$ respectively. In other words, $\mathrm{T}'(x)$ is the best linear approximation of $\mathrm{T}$ near $x$.

Now apart from those mathematical technicalities, Newton-Kantorovich has in essence the same structure as that of the original Newton's method. That is, it applies the following sequence of steps:

1. **Linearize the Operator**:
   Given an approximation $ x_n $, we consider the Fréchet derivative of $ \mathrm{T} $, denoted by $ \mathrm{T}'(x_n) $. This derivative is a linear operator that provides a local approximation of  $ \mathrm{T} $, near $ x_n $.

2. **Set Up the Newton Step**:
   The method then solves the linearized equation for a correction $ h_n $:

   $$
   \mathrm{T}(x_n) h_n = \mathrm{T}(x_n) - x_n.
   $$
   This equation represents a linear system where $ h_n $ is chosen to minimize the difference between $ x_n $ and $ \mathrm{T}(x_n) $ with respect to the operator's local behavior.

3. **Update the Solution**:
   The new approximation $ x_{n+1} $ is then given by:

   $$
   x_{n+1} = x_n - h_n.
   $$
   This correction step refines $ x_n $, bringing it closer to the true solution.

4. **Repeat Until Convergence**:
   We repeat the linearization and update steps until the solution $ x_n $ converges to the desired tolerance, which can be verified by checking that $ \|\mathrm{T}(x_n) - x_n\| $ is sufficiently small, or by monitoring the norm $ \|x_{n+1} - x_n\| $.

The convergence of Newton-Kantorovich does not hinge on $ \mathrm{T} $ being a contraction over the entire domain -- as it could be the case for successive approximation. The convergence properties of the Newton-Kantorovich method are as follows:

1. **Local Convergence**: Under mild conditions (e.g., $\mathrm{T}$ is Fréchet differentiable and $\mathrm{T}'(x)$ is invertible near the solution), the method converges locally. This means that if the initial guess is sufficiently close to the true solution, the method will converge.

2. **Global Convergence**: Global convergence is not guaranteed in general. However, under stronger conditions (e.g.,  $ \mathrm{T} $, is analytic and satisfies certain bounds), the method can converge globally.

3. **Rate of Convergence**: When the method converges, it typically exhibits quadratic convergence. This means that the error at each step is proportional to the square of the error at the previous step:

   $$
   \|x_{n+1} - x^*\| \leq C\|x_n - x^*\|^2
   $$

   where $x^*$ is the true solution and $C$ is some constant. This quadratic convergence is significantly faster than the linear convergence typically seen in methods like successive approximation.

## Optimality Equations for Infinite-Horizon MDPs

Recall that in the finite-horizon setting, the optimality equations are:

$$
v_n(s) = \max_{a \in A_s} \left\{r(s, a) + \sum_{j \in S} \gamma p(j | s, a) v_{n+1}(j)\right\}
$$

Intuitively, we would expect that by taking the limit of $n$ to infity, we might get the nonlinear equations: 

$$
v(s) = \max_{a \in A_s} \left\{r(s, a) + \sum_{j \in S} \gamma p(j | s, a) v(j)\right\}
$$

which are called the optimality equations or Bellman equations for infinite-horizon MDPs.
We can also adopt an operator-theoretic perspective and define a (nonlinear) operator $\mathrm{L}$ on the space $V$ 
of bounded real-valued functions on the state space $S$:

$$
\mathrm{L} \mathbf{v} \equiv \max_{d \in D^{MD}} \left\{\mathbf{r}_d + \gamma \mathbf{P}_d \mathbf{v}\right\}
$$
