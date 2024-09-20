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

# Continuous-Time Dynamical Systems

## State-Space Models 

We extend our focus from the discrete-time setting to trajectory optimization in continuous time. Such models are omnipresent in various branches of science and engineering, where the dynamics of physical, biological, or economic systems are often described in terms of continuous-time differential equations. Here, we consider models given by ordinary differential equations (ODEs). However, continuous-time optimal control methods also exist beyond ODEs; for example, using stochastic differential equations (SDEs) or partial differential equations (PDEs).

An example of such state-space representation is the following:

\begin{align*}
\dot{\mathbf{x}}(t) &= \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)) \\
\mathbf{y}(t) &= \mathbf{h}(\mathbf{x}(t), \mathbf{u}(t))
\end{align*}

The function $\mathbf{f}$ describes the dynamics of the system, giving the rate of change of the state vector as a function of the current state $\mathbf{x}(t)$ and control input $\mathbf{u}(t)$. The function $\mathbf{h}$ is the output mapping, which determines the measured output $\mathbf{y}(t)$ based on the current state $\mathbf{x}(t)$ and control input $\mathbf{u}(t)$. This state-space representation is reminiscent of recurrent neural networks (RNNs), albeit in discrete time, where we maintain a hidden state and then map it to an observable space.

A state space model is called a linear state space model (or simply a linear system) if the functions $\mathbf{f}$ and $\mathbf{h}$ are linear in $\mathbf{x}(t)$ and $\mathbf{u}(t)$. In this case, the model can be written as:

\begin{align*}
\dot{\mathbf{x}}(t) &= \mathbf{A}\mathbf{x}(t) + \mathbf{B}\mathbf{u}(t) \\
\mathbf{y}(t) &= \mathbf{C}\mathbf{x}(t) + \mathbf{D}\mathbf{u}(t)
\end{align*}

where $\mathbf{A}$, $\mathbf{B}$, $\mathbf{C}$, and $\mathbf{D}$ are constant matrices. The matrix $\mathbf{A}$ is called the dynamics matrix, $\mathbf{B}$ is the control matrix, $\mathbf{C}$ is the sensor matrix, and $\mathbf{D}$ is the direct term matrix. If the model does not have a direct term, it means that the control input $\mathbf{u}(t)$ does not directly influence the output $\mathbf{y}(t)$.

It is worth noting that linear models like the one presented above are becoming increasingly popular thanks to the development of structured state space models (S4 and such) [Gu et al., 2022]. These models leverage the inherent structure and properties of linear systems to design more efficient and interpretable neural networks for processing sequential data.

## Numerical Methods for Solving ODEs

An ODE is an implicit representation of a state-space trajectory: it tells us how the state changes in time but not precisely what the state is at any given time. To find out this information, we need to either solve the ODE analytically (for some special structure) or, as we're going to do, solve them numerically. This numerical procedure is meant to solve what is called an IVP (initial value problem) of the form:

$$
\text{Find } x(t) \text{ given } \dot{x}(t) = f(x(t), t) \text{ and } x(t_0) = x_0
$$

### Euler's Method 

The algorithm to solve this problem is, in its simplest form, a for loop which closely resembles the updates encountered in gradient descent (in fact, gradient descent can be derived from the gradient flow ODE, but that's another discussion). The so-called explicit Euler's method can be implemented as follow: 

````{prf:algorithm} Euler's method
:label: euler-method

**Input:** $f(x, t)$, $x_0$, $t_0$, $t_{end}$, $h$

**Output:** Approximate solution $x(t)$ at discrete time points

1. Initialize $t = t_0$, $x = x_0$
2. While $t < t_{end}$:
3.   Compute $x_{new} = x + h f(x, t)$
4.   Update $t = t + h$
5.   Update $x = x_{new}$
6.   Store or output the pair $(t, x)$
7. End While
````

Consider the following simple dynamical system of a ballistic motion model, neglecting air resistance. The state of the system is described by two variables: $y(t)$: vertical position at time $t$ and $v(t)$, the vertical velocity at time $t$. The corresponding ODE is: 

$$
\begin{aligned}
\frac{dy}{dt} &= v \\
\frac{dv}{dt} &= -g
\end{aligned}
$$

where $g \approx 9.81 \text{ m/s}^2$ is the acceleration due to gravity. In our code, we use the initial conditions 
$y(0) = 0 \text{ m}$ and $v(0) = v_0 \text{ m/s}$ where $v_0$ is the initial velocity (in this case, $v_0 = 20 \text{ m/s}$). 
The analytical solution to this system is:

$$
\begin{aligned}
y(t) &= v_0t - \frac{1}{2}gt^2 \\
v(t) &= v_0 - gt
\end{aligned}
$$

This system models the vertical motion of an object launched upward, reaching a maximum height before falling back down due to gravity.

Euler's method can be obtained by taking the first-order Taylor expansion of $x(t)$ at $t$:

$$
x(t + h) \approx x(t) + h \frac{dx}{dt}(t) = x(t) + h f(x(t), t)
$$

Each step of the algorithm therefore involves approximating the function with a linear function of slope $f$ over the given interval $h$. 

```{code-cell} ipython3
:tags: [hide-input]
:load: code/euler_step_size_viz.py
```

Another way to understand Euler's method is through the fundamental theorem of calculus:

$$
x(t + h) = x(t) + \int_t^{t+h} f(x(\tau), \tau) d\tau
$$

We then approximate the integral term with a box of width $h$ and height $f$, and therefore of area $h f$.
```{code-cell} ipython3
:tags: [hide-input]
:load: code/euler_integral_approximation_viz.py
```


### Implicit Euler's Method

An alternative approach is the Implicit Euler method, also known as the Backward Euler method. Instead of using the derivative at the current point to step forward, it uses the derivative at the end of the interval. This leads to the following update rule:

$$
x_{new} = x + h f(x_{new}, t_{new})
$$

Note that $x_{new}$ appears on both sides of the equation, making this an implicit method. The algorithm for the Implicit Euler method can be described as follows:

````{prf:algorithm} Implicit Euler's Method
:label: implicit-euler-method

**Input:** $f(x, t)$, $x_0$, $t_0$, $t_{end}$, $h$

**Output:** Approximate solution $x(t)$ at discrete time points

1. Initialize $t = t_0$, $x = x_0$
2. While $t < t_{end}$:
3.   Set $t_{new} = t + h$
4.   Solve for $x_{new}$ in the equation: $x_{new} = x + h f(x_{new}, t_{new})$
5.   Update $t = t_{new}$
6.   Update $x = x_{new}$
7.   Store or output the pair $(t, x)$
8. End While
````

The key difference in the Implicit Euler method is step 4, where we need to solve a (potentially nonlinear) equation to find $x_{new}$. This is typically done using iterative methods such as fixed-point iteration or Newton's method.

#### Stiff ODEs

While the Implicit Euler method requires more computation per step, it often allows for larger step sizes and can provide better stability for certain types of problems, especially stiff ODEs. 

Stiff ODEs are differential equations for which certain numerical methods for solving the equation are numerically unstable, unless the step size is taken to be extremely small. These ODEs typically involve multiple processes occurring at widely different rates. In a stiff problem, the fastest-changing component of the solution can make the numerical method unstable unless the step size is extremely small. However, such a small step size may lead to an impractical amount of computation to traverse the entire interval of interest.

For example, consider a chemical reaction where some reactions occur very quickly while others occur much more slowly. The fast reactions quickly approach their equilibrium, but small perturbations in the slower reactions can cause rapid changes in the fast reactions. 

A classic example of a stiff ODE is the Van der Pol oscillator with a large parameter. The Van der Pol equation is:

$$
\frac{d^2x}{dt^2} - \mu(1-x^2)\frac{dx}{dt} + x = 0
$$

where $\mu$ is a scalar parameter. This second-order ODE can be transformed into a system of first-order ODEs by introducing a new variable $y = \frac{dx}{dt}$:

$$
\begin{aligned}
\frac{dx}{dt} &= y \\
\frac{dy}{dt} &= \mu(1-x^2)y - x
\end{aligned}
$$

When $\mu$ is large (e.g., $\mu = 1000$), this system becomes stiff. The large $\mu$ causes rapid changes in $y$ when $x$ is near ±1, but slower changes elsewhere. This leads to a solution with sharp transitions followed by periods of gradual change.

### Trapezoid Method 

The trapezoid method, also known as the trapezoidal rule, offers improved accuracy and stability compared to the simple Euler method. The name "trapezoid method" comes from the idea of using a trapezoid to approximate the integral term in the fundamental theorem of calculus. This leads to the following update rule:

$$
x_{new} = x + \frac{h}{2}[f(x, t) + f(x_{new}, t_{new})]
$$

where $ t_{new} = t + h $. Note that this formula involves $ x_{new} $ on both sides of the equation, making it an implicit method, similar to the implicit Euler method discussed earlier.

```{code-cell} ipython3
:tags: [hide-input]
:load: code/trapezoid_integral_approximation_viz.py
```

Algorithmically, the trapezoid method can be described as follows:

````{prf:algorithm} Trapezoid Method

**Input:** $ f(x, t) $, $ x_0 $, $ t_0 $, $ t_{end} $, $ h $

**Output:** Approximate solution $ x(t) $ at discrete time points

1. Initialize $ t = t_0 $, $ x = x_0 $
2. While $ t < t_{end} $:
3. &emsp; Set $ t_{new} = t + h $
4. &emsp; Solve for $ x_{new} $in the equation: $ x_{new} = x + \frac{h}{2}[f(x, t) + f(x_{new}, t_{new})] $
5. &emsp; Update $ t = t_{new} $
6. &emsp; Update $ x = x_{new} $
7. &emsp; Store or output the pair $ (t, x) $
````

The trapezoid method can also be derived by averaging the forward Euler and backward Euler methods. Recall that:

1. **Forward Euler method:**

   $$ x_{n+1} = x_n + h f(x_n, t_n) $$

2. **Backward Euler method:**

   $$ x_{n+1} = x_n + h f(x_{n+1}, t_{n+1}) $$

Taking the average of these two methods yields:

$$
\begin{aligned}
x_{n+1} &= \frac{1}{2} \left( x_n + h f(x_n, t_n) \right) + \frac{1}{2} \left( x_n + h f(x_{n+1}, t_{n+1}) \right) \\
&= x_n + \frac{h}{2} \left( f(x_n, t_n) + f(x_{n+1}, t_{n+1}) \right)
\end{aligned}
$$

This is precisely the update rule for the trapezoid method. Recall that the forward Euler method approximates the solution by extrapolating linearly using the slope at the beginning of the interval $[t_n, t_{n+1}] $. In contrast, the backward Euler method extrapolates linearly using the slope at the end of the interval. The trapezoid method, on the other hand, averages these two slopes. This averaging provides better approximation properties than either of the methods alone, offering both stability and accuracy. Note finally that unlike the forward or backward Euler methods, the trapezoid method is also symmetric in time. This means that if you were to reverse time and apply the method backward, you would get the same results (up to numerical precision). 


### Trapezoidal Predictor-Corrector

The trapezoid method can also be implemented under the so-called predictor-corrector framework. This interpretation reformulates the implicit trapezoid rule into an explicit two-step process:

1. **Predictor Step**:  
   We make an initial guess for $ x_{n+1} $ using the forward Euler method:

   $$
   x_{n+1}^* = x_n + h f(x_n, t_n)
   $$

   This is our "predictor" step, where $ x_{n+1}^* $ is the predicted value of $ x_{n+1} $.

2. **Corrector Step**:  
   We then use this predicted value to estimate $ f(x_{n+1}^*, t_{n+1}) $ and apply the trapezoid formula:

   $$
   x_{n+1} = x_n + \frac{h}{2} \left[ f(x_n, t_n) + f(x_{n+1}^*, t_{n+1}) \right]
   $$

   This is our "corrector" step, where the initial guess $ x_{n+1}^* $ is corrected by taking into account the slope at $ (x_{n+1}^*, t_{n+1}) $.

This two-step process is similar to performing one iteration of Newton's method to solve the implicit trapezoid equation, starting from the Euler prediction. However, to fully solve the implicit equation, multiple iterations would be necessary until convergence is achieved.

```{code-cell} ipython3
:tags: [hide-input]
:load: code/predictor_corrector_trapezoid_viz.py
```


# Trajectory Optimization in Continuous Time

As studied earlier in the discrete-time setting, we consider three variants of the continuous-time optimal control problem (COCP) with path constraints and bounds:

::::{grid}
:gutter: 1

:::{grid-item}
````{prf:definition} Mayer Problem
$$
\begin{aligned}
    \text{minimize} \quad & c(\mathbf{x}(t_f)) \\
    \text{subject to} \quad & \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)) \\
                            & \mathbf{g}(\mathbf{x}(t), \mathbf{u}(t)) \leq \mathbf{0} \\
                            & \mathbf{x}_{\text{min}} \leq \mathbf{x}(t) \leq \mathbf{x}_{\text{max}} \\
                            & \mathbf{u}_{\text{min}} \leq \mathbf{u}(t) \leq \mathbf{u}_{\text{max}} \\
    \text{given} \quad & \mathbf{x}(t_0) = \mathbf{x}_0 \enspace .
\end{aligned}
$$

````
:::

:::{grid-item}
````{prf:definition} Lagrange Problem
$$
\begin{aligned}
    \text{minimize} \quad & \int_{t_0}^{t_f} c(\mathbf{x}(t), \mathbf{u}(t)) \, dt \\
    \text{subject to} \quad & \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)) \\
                            & \mathbf{g}(\mathbf{x}(t), \mathbf{u}(t)) \leq \mathbf{0} \\
                            & \mathbf{x}_{\text{min}} \leq \mathbf{x}(t) \leq \mathbf{x}_{\text{max}} \\
                            & \mathbf{u}_{\text{min}} \leq \mathbf{u}(t) \leq \mathbf{u}_{\text{max}} \\
    \text{given} \quad & \mathbf{x}(t_0) = \mathbf{x}_0 \enspace .
\end{aligned}
$$

````
:::

:::{grid-item}
````{prf:definition} Bolza Problem
$$
\begin{aligned}
    \text{minimize} \quad & c(\mathbf{x}(t_f)) + \int_{t_0}^{t_f} c(\mathbf{x}(t), \mathbf{u}(t)) \, dt \\
    \text{subject to} \quad & \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)) \\
                            & \mathbf{g}(\mathbf{x}(t), \mathbf{u}(t)) \leq \mathbf{0} \\
                            & \mathbf{x}_{\text{min}} \leq \mathbf{x}(t) \leq \mathbf{x}_{\text{max}} \\
                            & \mathbf{u}_{\text{min}} \leq \mathbf{u}(t) \leq \mathbf{u}_{\text{max}} \\
    \text{given} \quad & \mathbf{x}(t_0) = \mathbf{x}_0 \enspace .
\end{aligned}
$$

````
:::
::::


In these formulations, the additional constraints are:

- Path constraints: $\mathbf{g}(\mathbf{x}(t), \mathbf{u}(t)) \leq \mathbf{0}$, which represent constraints that must be satisfied at all times along the trajectory.
- State bounds: $\mathbf{x}_{\text{min}} \leq \mathbf{x}(t) \leq \mathbf{x}_{\text{max}}$, which specify the lower and upper bounds on the state variables.
- Control bounds: $\mathbf{u}_{\text{min}} \leq \mathbf{u}(t) \leq \mathbf{u}_{\text{max}}$, which specify the lower and upper bounds on the control inputs.

Furthermore, we may also encounter variations of the above problems under the assumption that horizon is infinite. For example: 

````{prf:definition} Infinite-Horizon Trajectory Optimization 
\begin{align*}
    &\text{minimize} \quad \int_{t_0}^{\infty} e^{-\rho t} c(\mathbf{x}(t), \mathbf{u}(t)) \, dt \\
    &\text{subject to} \quad \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)) \\
    &\phantom{\text{subject to}} \quad \mathbf{g}(\mathbf{x}(t), \mathbf{u}(t)) \leq \mathbf{0} \\
    &\phantom{\text{subject to}} \quad \mathbf{x}_{\text{min}} \leq \mathbf{x}(t) \leq \mathbf{x}_{\text{max}} \\
    &\phantom{\text{subject to}} \quad \mathbf{u}_{\text{min}} \leq \mathbf{u}(t) \leq \mathbf{u}_{\text{max}} \\
    &\text{given} \quad \mathbf{x}(t_0) = \mathbf{x}_0 \enspace .
\end{align*}
````
````{margin}
```{note}
In optimal control problems, the use of an **exponential discount rate** $ e^{-\rho t} $ is favored over a discrete-time **discount factor** $ \gamma $ (commonly used in reinforcement learning). For small time steps $ \Delta t $, we can approximate the exponential discounting as
     $e^{-\rho \Delta t} \approx 1 - \rho \Delta t.$ Therefore, as $ \Delta t \to 0 $, the continuous discount factor $ e^{-\rho t} $ corresponds to a discrete discount factor $ \gamma = e^{-\rho \Delta t} \approx 1 - \rho \Delta t $. 
```
````
In this formulation, the term $e^{-\rho t}$ is a discount factor that exponentially decreases the importance of future costs relative to the present. The parameter $ \rho > 0$ is the discount rate. A larger value of $ \rho $ places more emphasis on the immediate cost and diminishes the impact of future costs. In infinite-horizon problems, the integral of the cost function $ \int_{t_0}^{\infty} c(\mathbf{x}(t), \mathbf{u}(t)) \, dt $ could potentially diverge because the cost accumulates over an infinite time period. Introducing the exponential term $ e^{-\rho t} $ guarantees that the integral converges as long as $ c(\mathbf{x}(t), \mathbf{u}(t)) $ grows at a slower rate than $ e^{\rho t} $. 


## Example Problems
### Inverted Pendulum

The inverted pendulum is a classic problem in control theory and robotics that demonstrates the challenge of stabilizing a dynamic system that is inherently unstable. The objective is to keep a pendulum balanced in the upright position by applying a control force, typically at its base. This setup is analogous to balancing a broomstick on your finger: any deviation from the vertical position will cause the system to tip over unless you actively counteract it with appropriate control actions.

We typically assume that the pendulum is mounted on a cart or movable base, which can move horizontally. The system's state is then characterized by four variables:

1. **Cart position**: $ x(t) $ — the horizontal position of the base.
2. **Cart velocity**: $ \dot{x}(t) $ — the speed of the cart.
3. **Pendulum angle**: $ \theta(t) $ — the angle between the pendulum and the vertical upright position.
4. **Angular velocity**: $ \dot{\theta}(t) $ — the rate at which the pendulum's angle is changing.

This setup is more complex because the controller must deal with interactions between two different types of motion: linear (the cart) and rotational (the pendulum). This system is said to be "underactuated" because the number of control inputs (one) is less than the number of state variables (four). This makes the problem more challenging and interesting from a control perspective.

We can simplify the problem by assuming that the base of the pendulum is fixed.  This is akin to having the bottom of the stick attached to a fixed pivot on a table. You can't move the base anymore; you can only apply small nudges at the pivot point to keep the stick balanced upright. In this case, you're only focusing on adjusting the stick's tilt without worrying about moving the base. This reduces the problem to stabilizing the pendulum’s upright orientation using only the rotational dynamics. The system's state can now be described by just two variables:

1. **Pendulum angle**: $ \theta(t) $ — the angle of the pendulum from the upright vertical position.
2. **Angular velocity**: $ \dot{\theta}(t) $ — the rate at which the pendulum's angle is changing.

The evolution of these two varibles is governed by the following ordinary differential equation:

\begin{equation}
\begin{bmatrix} \dot{\theta}(t) \\ \ddot{\theta}(t) \end{bmatrix} = \begin{bmatrix} \dot{\theta}(t) \\ \frac{mgl}{J_t} \sin{\theta(t)} - \frac{\gamma}{J_t} \dot{\theta}(t) + \frac{l}{J_t} u(t) \cos{\theta(t)} \end{bmatrix}, \quad y(t) = \theta(t)
\end{equation}

where:
- $m$ is the mass of the pendulum
- $g$ is the acceleration due to gravity
- $l$ is the length of the pendulum
- $\gamma$ is the coefficient of rotational friction
- $J_t = J + ml^2$ is the total moment of inertia, with $J$ being the pendulum's moment of inertia about its center of mass
- $u(t)$ is the control force applied at the base
- $y(t) = \theta(t)$ is the measured output (the pendulum's angle)

We expect that when no control is applied to the system, the rod should be falling down when started from the upright position. 

```{code-cell} ipython3
:tags: [hide-input]
:load: code/pendulum.py
```

#### Looking Under the Hood: Pendulum in the Gym Environment

Gym is a widely used abstraction layer for defining discrete-time reinforcement learning problems. In reinforcement learning research, there's often a desire to develop general-purpose algorithms that are problem-agnostic. This research mindset leads us to voluntarily avoid considering the implementation details of a given environment. While this approach is understandable from a research perspective, it may not be optimal from a pragmatic, solution-driven standpoint where we care about solving specific problems efficiently. If we genuinely wanted to solve this problem without prior knowledge, why not look under the hood and embrace its nature as a trajectory optimization problem?

Let's examine the code and reverse-engineer the original continuous-time problem hidden behind the abstraction layer. Although the pendulum problem may have limited practical relevance as a real-world application, it serves as an excellent example for our analysis. In the current version of [Pendulum](https://github.com/openai/gym/blob/dcd185843a62953e27c2d54dc8c2d647d604b635/gym/envs/classic_control/pendulum.py), we find that the Gym implementation uses a simplified model. Like our implementation, it assumes a fixed base and doesn't model cart movement. The state is also represented by the pendulum angle and angular velocity.
However, the equations of motion implemented in the Gym environment are different and correspond to the following ODE:

\begin{align*}
\dot{\theta} &= \theta_{dot} \\
\dot{\theta}_{dot} &= \frac{3g}{2l} \sin(\theta) + \frac{3}{ml^2} u
\end{align*}

Compared to our simplified model, the Gym implementation makes the following additional assumptions:

1. It omits the term $\frac{\gamma}{J_t} \dot{\theta}(t)$, which represents damping or air resistance. This means that it assumes an idealized pendulum that doesn't naturally slow down over time. 

2. It uses $ml^2$ instead of $J_t = J + ml^2$, which assumes that all mass is concentrated at the pendulum's end (like a point mass on a massless rod), rather than accounting for mass distribution along the pendulum. 

3. The control input $u$ is applied directly, without a $\cos \theta(t)$ term, which means that the applied torque has the same effect regardless of the pendulum's position, rather than varying with angle. For example, imagine trying to push a door open. When the door is almost closed (pendulum near vertical), a small push perpendicular to the door (analogous to our control input) can easily start it moving. However, when the door is already wide open (pendulum horizontal), the same push has little effect on the door's angle. In a more detailed model, this would be captured by the $\cos \theta(t)$ term, which is maximum when the pendulum is vertical ($\cos 0° = 1$) and zero when horizontal ($\cos 90° = 0$).

The goal remains to stabilize the rod upright, but the way in which this encoded is through the following instantenous cost function:

\begin{align*}
c(\theta, \dot{\theta}, u) &= (\text{normalize}(\theta))^2 + 0.1\dot{\theta}^2 + 0.001u^2\\
\text{normalize}(\theta) &= ((\theta + \pi) \bmod 2\pi) - \pi
\end{align*}

This cost function penalizes deviations from the upright position (first term), discouraging rapid motion (second term), and limiting control effort (third term). The relative weights has been manually chosen to balance the primary goal of upright stabilization with the secondary aims of smooth motion and energy efficiency. The normalization ensures that the angle is always in the range $[-\pi, \pi]$ so that the pendulum positions (e.g., $0$ and $2\pi$) are treated identically, which could otherwise confuse learning algorithms.

Studying the code further, we find that it imposes bound constraints on both the control input and the angular velocity through clipping operations:

\begin{align*}
u &= \max(\min(u, u_{max}), -u_{max}) \\
\dot{\theta} &= \max(\min(\dot{\theta}, \dot{\theta}_{max}), -\dot{\theta}_{max})
\end{align*}

Where $u_{max} = 2.0$ and $\dot{\theta}_{max} = 8.0$.  Finally, when inspecting the [`step`](https://github.com/openai/gym/blob/dcd185843a62953e27c2d54dc8c2d647d604b635/gym/envs/classic_control/pendulum.py#L133) function, we find that the dynamics are discretized using forward Euler under a fixed step size of $h=0.0.5$. Overall, the discrete-time trajectory optimization problem implemented in Gym is the following: 
\begin{align*}
\min_{u_k} \quad & J = \sum_{k=0}^{N-1} c(\theta_k, \dot{\theta}_k, u_k) \\
\text{subject to:} \quad & \theta_{k+1} = \theta_k + \dot{\theta}_k \cdot h \\
& \dot{\theta}_{k+1} = \dot{\theta}_k + \left(\frac{3g}{2l}\sin(\theta_k) + \frac{3}{ml^2}u_k\right) \cdot h \\
& -u_{\max} \leq u_k \leq u_{\max} \\
& -\dot{\theta}_{\max} \leq \dot{\theta}_k \leq \dot{\theta}_{\max}, \quad k = 0, 1, ..., N-1 \\
\text{given:} \quad      & \theta_0 = \theta_{\text{initial}}, \quad \dot{\theta}_0 = \dot{\theta}_{\text{initial}}, \quad N = 200
\end{align*}

 with $g = 10.0$, $l = 1.0$, $m = 1.0$, $u_{max} = 2.0$, and $\dot{\theta}_{max} = 8.0$. This discrete-time problem corresponds to the following continuous-time optimal control problem:

\begin{align*}
\min_{u(t)} \quad & J = \int_{0}^{T} c(\theta(t), \dot{\theta}(t), u(t)) dt \\
\text{subject to:} \quad & \dot{\theta}(t) = \dot{\theta}(t) \\
& \ddot{\theta}(t) = \frac{3g}{2l}\sin(\theta(t)) + \frac{3}{ml^2}u(t) \\
& -u_{\max} \leq u(t) \leq u_{\max} \\
& -\dot{\theta}_{\max} \leq \dot{\theta}(t) \leq \dot{\theta}_{\max} \\
\text{given:} \quad      & \theta(0) = \theta_0, \quad \dot{\theta}(0) = \dot{\theta}_0, \quad T = 10 \text{ seconds}
\end{align*}


### Heat Exchanger 

![Heat Exchanger](_static/heat_exchanger.svg)


We are considering a system where fluid flows through a tube, and the goal is to control the temperature of the fluid by adjusting the temperature of the tube's wall over time. The wall temperature, denoted as $ T_w(t) $, can be changed as a function of time, but it remains the same along the length of the tube. On the other hand, the temperature of the fluid inside the tube, $ T(z, t) $, depends both on its position along the tube $ z $ and on time $ t $. It evolves according to the following partial differential equation:

$$
\frac{\partial T}{\partial t} = -v \frac{\partial T}{\partial z} + \frac{h}{\rho C_p} (T_w(t) - T)
$$

where we have:
- $ v $: the average speed of the fluid moving through the tube,
- $ h $: how easily heat transfers from the wall to the fluid,
- $ \rho $ and $ C_p $: the fluid’s density and heat capacity.

This equation describes how the fluid's temperature changes as it moves along the tube and interacts with the tube's wall temperature. The fluid enters the tube with an initial temperature $ T_0 $ at the inlet (where $ z = 0 $). Our objective is to adjust the wall temperature $ T_w(t) $ so that by a specific final time $ t_f $, the fluid's temperature reaches a desired distribution $ T_s(z) $ along the length of the tube. The relationship for $ T_s(z) $ under steady-state conditions (ie. when changes over time are no longer considered), is given by:

$$
\frac{d T_s}{d z} = \frac{h}{v \rho C_p}[\theta - T_s]
$$

where $ \theta $ is a constant temperature we want to maintain at the wall. The objective is to control the wall temperature $ T_w(t) $ so that by the end of the time interval $ t_f $, the fluid temperature $ T(z, t_f) $ is as close as possible to the desired distribution $ T_s(z) $. This can be formalized by minimizing the following quantity:

$$
I = \int_0^L \left[T(z, t_f) - T_s(z)\right]^2 dz
$$

where $ L $ is the length of the tube. Additionally, we require that the wall temperature cannot exceed a maximum allowable value $ T_{\max} $:

$$
T_w(t) \leq T_{\max}
$$


### Nuclear Reactor

![Nuclear Reactor Diagram](_static/nuclear_reactor.svg)

In a nuclear reactor, neutrons interact with fissile nuclei, causing nuclear fission. This process produces more neutrons and smaller fissile nuclei called precursors. The precursors subsequently absorb more neutrons, generating "delayed" neutrons. The kinetic energy of these products is converted into thermal energy through collisions with neighboring atoms. The reactor's power output is determined by the concentration of neutrons available for nuclear fission.

The reaction kinetics can be modeled using a system of ordinary differential equations:

\begin{align*}
\dot{x}(t) &= \frac{r(t)x(t) - \alpha x^2(t) - \beta x(t)}{\tau} + \mu y(t), & x(0) &= x_0 \\
\dot{y}(t) &= \frac{\beta x(t)}{\tau} - \mu y(t), & y(0) &= y_0
\end{align*}

where:
- $x(t)$: concentration of neutrons at time $t$
- $y(t)$: concentration of precursors at time $t$
- $t$: time
- $r(t) = r[u(t)]$: degree of change in neutron multiplication at time $t$ as a function of control rod displacement $u(t)$
- $\alpha$: reactivity coefficient
- $\beta$: fraction of delayed neutrons
- $\mu$: decay constant for precursors
- $\tau$: average time taken by a neutron to produce a neutron or precursor

The power output can be adjusted based on demand by inserting or retracting a neutron-absorbing control rod. Inserting the control rod absorbs neutrons, reducing the heat flux and power output, while retracting the rod has the opposite effect.

The objective is to change the neutron concentration $x(t)$ from an initial value $x_0$ to a stable value $x_\mathrm{f}$ at time $t_\mathrm{f}$ while minimizing the displacement of the control rod. This can be formulated as an optimal control problem, where the goal is to find the control function $u(t)$ that minimizes the objective functional:

\begin{equation*}
I = \int_0^{t_\mathrm{f}} u^2(t) \, \mathrm{d}t
\end{equation*}

subject to the final conditions:

\begin{align*}
x(t_\mathrm{f}) &= x_\mathrm{f} \\
\dot{x}(t_\mathrm{f}) &= 0
\end{align*}

and the constraint $|u(t)| \leq u_\mathrm{max}$

### Chemotherapy

Chemotherapy uses drugs to kill cancer cells. However, these drugs can also have toxic effects on healthy cells in the body. To optimize the effectiveness of chemotherapy while minimizing its side effects, we can formulate an optimal control problem. 

The drug concentration $y_1(t)$ and the number of immune cells $y_2(t)$, healthy cells $y_3(t)$, and cancer cells $y_4(t)$ in an organ at any time $t$ during chemotherapy can be modeled using a system of ordinary differential equations:

\begin{align*}
\dot{y}_1(t) &= u(t) - \gamma_6 y_1(t) \\
\dot{y}_2(t) &= \dot{y}_{2,\text{in}} + r_2 \frac{y_2(t) y_4(t)}{\beta_2 + y_4(t)} - \gamma_3 y_2(t) y_4(t) - \gamma_4 y_2(t) - \alpha_2 y_2(t) \left(1 - e^{-y_1(t) \lambda_2}\right) \\
\dot{y}_3(t) &= r_3 y_3(t) \left(1 - \beta_3 y_3(t)\right) - \gamma_5 y_3(t) y_4(t) - \alpha_3 y_3(t) \left(1 - e^{-y_1(t) \lambda_3}\right) \\
\dot{y}_4(t) &= r_1 y_4(t) \left(1 - \beta_1 y_4(t)\right) - \gamma_1 y_3(t) y_4(t) - \gamma_2 y_2(t) y_4(t) - \alpha_1 y_4(t) \left(1 - e^{-y_1(t) \lambda_1}\right)
\end{align*}

where:
- $y_1(t)$: drug concentration in the organ at time $t$
- $y_2(t)$: number of immune cells in the organ at time $t$
- $y_3(t)$: number of healthy cells in the organ at time $t$
- $y_4(t)$: number of cancer cells in the organ at time $t$
- $\dot{y}_{2,\text{in}}$: constant rate of immune cells entering the organ to fight cancer cells
- $u(t)$: rate of drug injection into the organ at time $t$
- $r_i, \beta_i$: constants in the growth terms
- $\alpha_i, \lambda_i$: constants in the decay terms due to the action of the drug
- $\gamma_i$: constants in the remaining decay terms

The objective is to minimize the number of cancer cells $y_4(t)$ in a specified time $t_\mathrm{f}$ while using the minimum amount of drug to reduce its toxic effects. This can be formulated as an optimal control problem, where the goal is to find the control function $u(t)$ that minimizes the objective functional:

\begin{equation*}
I = y_4(t_\mathrm{f}) + \int_0^{t_\mathrm{f}} u(t) \, \mathrm{d}t
\end{equation*}

subject to the system dynamics, initial conditions, and the constraint $u(t) \geq 0$.

Additional constraints may include:
- Maintaining a minimum number of healthy cells during treatment:
  \begin{equation*}
  y_3(t) \geq y_{3,\min}
  \end{equation*}
- Imposing an upper limit on the drug dosage:
  \begin{equation*}
  u(t) \leq u_\max
  \end{equation*}

### Government Corruption 

In this model from Feichtinger and Wirl (1994), we aim to understand the incentives for politicians to engage in corrupt activities or to combat corruption. The model considers a politician's popularity as a dynamic process that is influenced by the public's memory of recent and past corruption. The objective is to find conditions under which self-interested politicians would choose to be honest or dishonest.

The model introduces the following notation:

- $C(t)$: accumulated awareness (knowledge) of past corruption at time $t$
- $u(t)$: extent of corruption (politician's control variable) at time $t$
- $\delta$: rate of forgetting past corruption
- $P(t)$: politician's popularity at time $t$
- $g(P)$: growth function of popularity; $g''(P) < 0$
- $f(C)$: function measuring the loss of popularity caused by $C$; $f'(C) > 0$, $f''(C) \geq 0$
- $U_1(P)$: benefits associated with being popular; $U_1'(P) > 0$, $U_1''(P) \leq 0$
- $U_2(u)$: benefits resulting from bribery and fraud; $U_2'(u) > 0$, $U_2''(u) < 0$
- $r$: discount rate

The dynamics of the public's memory of recent and past corruption $C(t)$ are modeled as:

\begin{align*}
\dot{C}(t) &= u(t) - \delta C(t), \quad C(0) = C_0
\end{align*}

The evolution of the politician's popularity $P(t)$ is governed by:

\begin{align*}
\dot{P}(t) &= g(P(t)) - f(C(t)), \quad P(0) = P_0
\end{align*}

The politician's objective is to maximize the following objective:

\begin{equation*}
\int_0^{\infty} e^{-rt} [U_1(P(t)) + U_2(u(t))] \, \mathrm{d}t
\end{equation*}

subject to the dynamics of corruption awareness and popularity.

The optimal control problem can be formulated as follows:

\begin{align*}
\max_{u(\cdot)} \quad & \int_0^{\infty} e^{-rt} [U_1(P(t)) + U_2(u(t))] \, \mathrm{d}t \\
\text{s.t.} \quad & \dot{C}(t) = u(t) - \delta C(t), \quad C(0) = C_0 \\
& \dot{P}(t) = g(P(t)) - f(C(t)), \quad P(0) = P_0
\end{align*}

The state variables are the accumulated awareness of past corruption $C(t)$ and the politician's popularity $P(t)$. The control variable is the extent of corruption $u(t)$. The objective functional represents the discounted stream of benefits coming from being honest (popularity) and from being dishonest (corruption).

# Direct Transcription Methods
 
When transitioning from discrete-time optimal control to continuous-time, we encounter several new computational challenges:

1. The optimization variables are now functions, not just discrete sequences of values stored in an array: we seek $x(t)$ and $u(t)$, which are continuous functions of time.
2. Evaluating a candidate pair $x(t)$ and $u(t)$ involves integration: both for assessing the integral term in the objective of Lagrange or Bolza problems, as well as for the dynamics expressed as constraints.

These two problems are ones of representation (which we address through function approximation) and integration (which we address through numerical integration methods).

These two elements combined give us the blueprint for many approaches known under the umbrella of direct transcription methods. The idea is to take an original continuous-time optimal control problem, which is an infinite-dimensional optimization problem, and transform it into a finite approximation as a standard NLP, similar to those we have studied in discrete-time optimal control problems.

In control theory and many fields like aeronautics, aerospace, or chemical engineering, the parameterization of either the control or state functions is typically done via polynomials. However, the approach presented here is general and might be applicable to other function approximators, including neural networks.

## Example: Life-Cycle Model

The life-cycle model of consumption is a problem in economics regarding how individuals allocate resources over their lifetime, trading-off present and future consumption depending on their income and ability to save or borrow. This model demonstrates the idea that individuals tend to prefer stable consumption, even when their income varies: a behavior called "consumption smoothing". This problem can be represented mathematically as follows:

$$
\begin{align*}
\max_{c(t)} \quad & \int_0^T e^{-\rho t} u(c(t)) dt \\
\text{subject to} \quad & \dot{A}(t) = f(A(t)) + w(t) - c(t) \\
& A(0) = A(T) = 0
\end{align*}
$$

where $[0, T]$ is the lifespan of an individual. In this model, we typically use a Constant Relative Risk Aversion (CRRA) utility function, 
$u(c) = \frac{c^{1-\gamma}}{1-\gamma}$, where larger values of the parameter $\gamma$ encode a stronger preference for a stable consumption.

The budget constraint, $\dot{A}(t) = f(A(t)) + w(t) - c(t)$, describes the evolution of assets, $A(t)$. Here, $f(A(t))$ represents returns on investments, $w(t)$ is wage income, and $c(t)$ is consumption. The asset return function $f(A) = 0.03A$ models a constant 3\% return on investments.
In our specific implementation, the choice of the wage function $w(t) = 1 + 0.1t - 0.001t^2$ is meant to represent a career trajectory where income rises initially and then falls. The boundary conditions $A(0) = A(T) = 0$ finally encodes the fact that individuals start and end life with zero assets (ie. no inheritances).

### Single Shooting Solution
To solve this problem numerically, we parameterize the entire consumption path as a cubic polynomial turning our original problem into:

$$
\begin{align*}
\min_{\theta_0,\theta_1,\theta_2,\theta_3} \quad & \int_0^T e^{-\rho t} u(c(t)) dt \\
\text{subject to} \quad & \dot{A}(t) = f(A(t)) + w(t) - c(t) \\
& c(t) = \theta_0 + \theta_1t + \theta_2t^2 + \theta_3t^3 \\
& A(0) = A(T) = 0
\end{align*}
$$

We then transcribe the problem using a fourth-order Runge-Kutta method (RK4) to simulate the assets dynamics and 
we discretize integral cost function using: 

   $$\int_0^T e^{-\rho t} u(c(t)) dt \approx \sum_{i=0}^{N-1} e^{-\rho t_i} u(c(t_i)) \Delta t$$

Finally, we explicitly enforce the boundary condition $A(T) = 0$ as an equality constraint in our optimization problem, which we solve using `scipy.optimize.minimize`. 
When using single shooting to eliminate the dynamics constraints, we the obtain the following NLP:

$$
\begin{align*}
\min_{\theta_0,\theta_1,\theta_2,\theta_3} \quad & -\sum_{i=0}^{N-1} e^{-\rho t_i} u(c_i) \Delta t \\
\text{subject to} \quad & \Phi(\theta_0,\theta_1,\theta_2,\theta_3) = 0 \\
\text{where}\quad & c_i = \theta_0 + \theta_1t_i + \theta_2t_i^2 + \theta_3t_i^3, \quad i = 0, \ldots, N
\end{align*}
$$

and $\Phi$ is defined by:

$$
\begin{align*}
\Phi(\theta_0,\theta_1,\theta_2,\theta_3) &= \Psi_N \circ \Psi_{N-1} \circ \cdots \circ \Psi_1(0) \\
\Psi_i(A) &= A + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4) \\
k_1 &= f(A) + w(t_i) - c_i \\
k_2 &= f(A + \frac{\Delta t}{2}k_1) + w(t_i + \frac{\Delta t}{2}) - c_{i+1/2} \\
k_3 &= f(A + \frac{\Delta t}{2}k_2) + w(t_i + \frac{\Delta t}{2}) - c_{i+1/2} \\
k_4 &= f(A + \Delta t k_3) + w(t_i + \Delta t) - c_{i+1} \\
c_i &= \theta_0 + \theta_1t_i + \theta_2t_i^2 + \theta_3t_i^3 \\
c_{i+1/2} &= \theta_0 + \theta_1(t_i + \frac{\Delta t}{2}) + \theta_2(t_i + \frac{\Delta t}{2})^2 + \theta_3(t_i + \frac{\Delta t}{2})^3
\end{align*}
$$

```{code-cell} ipython3
:tags: [hide-input]
:load: code/life_cycle_rk4.py
```