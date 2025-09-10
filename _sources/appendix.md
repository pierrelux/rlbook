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



## Example COCPs

### Pendulum in the Gym Environment

<!-- Gym is a widely used abstraction layer for defining discrete-time reinforcement learning problems. In reinforcement learning research, there's often a desire to develop general-purpose algorithms that are problem-agnostic. This research mindset leads us to voluntarily avoid considering the implementation details of a given environment. While this approach is understandable from a research perspective, it may not be optimal from a pragmatic, solution-driven standpoint where we care about solving specific problems efficiently. If we genuinely wanted to solve this problem without prior knowledge, why not look under the hood and embrace its nature as a trajectory optimization problem? -->

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

## Solving Initial Value Problems

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
### Collocation Methods

The numerical integration methods we discussed earlier are inherently **sequential**: given an initial state, we step forward in time and approximate what happens over a short interval. The accuracy of this procedure depends on the chosen rule (Euler, trapezoid, Runge–Kutta) and on the information available locally. Each new state is obtained by evaluating a formula that approximates the derivative or integral over that small step.

**Collocation methods provide an alternative viewpoint.** Instead of advancing one step at a time, they approximate the entire trajectory with a finite set of basis functions and require the dynamics to hold at selected points. This replaces the original differential equation with a system of **algebraic equations**: relations among the coefficients of the basis functions that must all be satisfied simultaneously. Solving these equations fixes the whole trajectory in one computation.

Seen from this angle, integration rules, spline interpolation, quadrature, and collocation are all instances of the same principle: an infinite-dimensional problem is reduced to finitely many parameters linked by numerical rules. The difference is mainly in scope. Sequential integration advances the state forward one interval at a time, which makes it simple but prone to error accumulation. Collocation belongs to the class of **simultaneous methods** already introduced for DOCPs: the entire trajectory is represented at once, the dynamics are imposed everywhere in the discretization, and approximation error is spread across the horizon rather than accumulating step by step.

This global enforcement comes at a computational cost since the resulting algebraic system is larger and denser. However, the benefit is precisely the structural one we saw in simultaneous methods earlier: by exposing the coupling between states, controls, and dynamics explicitly, collocation allows solvers to exploit sparsity and to enforce path constraints directly at the collocation points. This is why collocation is especially effective for challenging continuous-time optimal control problems where robustness and constraint satisfaction are central.


### Quick Primer on Polynomials

Collocation methods are based on polynomial approximation theory. Therefore, the first step in developing collocation-based optimal control techniques is to review the fundamentals of polynomial functions. 

Polynomials are typically introduced through their standard form:

$$
p(t) = a_n t^n + a_{n-1} t^{n-1} + \cdots + a_1 t + a_0
$$

In this expression, the $a_i$ are coefficients which linearly combine the powers of $t$ to represent a function. The set of functions $\{ 1, t, t^2, t^3, \ldots, t^n \}$ used in the standard polynomial representation is called the **monomial basis**. 

In linear algebra, a basis is a set of vectors in a vector space such that any vector in the space can be uniquely represented as a linear combination of these basis vectors. In the same way, a **polynomial basis** is such that any function $ f(x) $ (within the function space) to be expressed as:

$$
f(x) = \sum_{k=0}^{\infty} c_k p_k(x),
$$

where the coefficients $ c_k $ are generally determined by solving a system of equation.

Just as vectors can be represented in different coordinate systems (bases), functions can also be expressed using various polynomial bases. However, the ability to apply a change of basis does not imply that all types of polynomials are equivalent from a practical standpoint. In practice, our choice of polynomial basis is dictated by considerations of efficiency, accuracy, and stability when approximating a function.

For instance, despite the monomial basis being easy to understand and implement, it often performs poorly in practice due to numerical instability. This instability arises as its coefficients take on large values: an ill-conditioning problem. The following kinds of polynomial often remedy this issues. 

#### Orthogonal Polynomials 

An **orthogonal polynomial basis** is a set of polynomials that are not only orthogonal to each other but also form a complete basis for a certain space of functions. This means that any function within that space can be represented as a linear combination of these polynomials. 

More precisely, let $ \{ p_0(x), p_1(x), p_2(x), \dots \} $ be a sequence of polynomials where each $ p_n(x) $ is a polynomial of degree $ n $. We say that this set forms an orthogonal polynomial basis if any polynomial $ q(x) $ of degree $ n $ or less can be uniquely expressed as a linear combination of $ \{ p_0(x), p_1(x), \dots, p_n(x) \} $. Furthermore, the orthogonality property means that for any $ i \neq j $:

$$
\langle p_i, p_j \rangle = \int_a^b p_i(x) p_j(x) w(x) \, dx = 0.
$$

for some weight function $ w(x) $ over a given interval of orthogonality $ [a, b] $. 

The orthogonality property allows to simplify the computation of the coefficients involved in the polynomial representation of a function. At a high level, what happens is that when taking the inner product of $ f(x) $ with each basis polynomial, $ p_k(x) $ isolates the corresponding coefficient $ c_k $, which can be found to be: 

$$
c_k = \frac{\langle f, p_k \rangle}{\langle p_k, p_k \rangle} = \frac{\int_a^b f(x) p_k(x) w(x) \, dx}{\int_a^b p_k(x)^2 w(x) \, dx}.
$$

Here are some examples of the most common orthogonal polynomials used in practice. 

##### Legendre Polynomials

Legendre polynomials $ \{ P_n(x) \} $ are defined on the interval $[-1, 1]$ and satisfy the orthogonality condition:

$$
\int_{-1}^{1} P_n(x) P_m(x) \, dx = 
\begin{cases}
0 & \text{if } n \neq m, \\
\frac{2}{2n + 1} & \text{if } n = m.
\end{cases}
$$

They can be generated using the recurrence relation:

$$
(n+1) P_{n+1}(x) = (2n + 1) x P_n(x) - n P_{n-1}(x),
$$

with initial conditions:

$$
P_0(x) = 1, \quad P_1(x) = x.
$$

The first four Legendre polynomials resulting from this recurrence are the following: 

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
from IPython.display import display, Math

def legendre_polynomial(n, x):
    if n == 0:
        return np.poly1d([1])
    elif n == 1:
        return x
    else:
        p0 = np.poly1d([1])
        p1 = x
        for k in range(2, n + 1):
            p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            p0, p1 = p1, p2
        return p1

def legendre_coefficients(n):
    x = np.poly1d([1, 0])  # Define a poly1d object to represent x
    poly = legendre_polynomial(n, x)
    return poly

def poly_to_latex(poly):
    coeffs = poly.coefficients
    variable = poly.variable
    
    terms = []
    for i, coeff in enumerate(coeffs):
        power = len(coeffs) - i - 1
        if coeff == 0:
            continue
        coeff_str = f"{coeff:.2g}" if coeff not in {1, -1} or power == 0 else ("-" if coeff == -1 else "")
        if power == 0:
            term = f"{coeff_str}"
        elif power == 1:
            term = f"{coeff_str}{variable}"
        else:
            term = f"{coeff_str}{variable}^{power}"
        terms.append(term)
    
    latex_poly = " + ".join(terms).replace(" + -", " - ")
    return latex_poly

for n in range(4):
    poly = legendre_coefficients(n)
    display(Math(f"P_{n}(x) = {poly_to_latex(poly)}"))
```

##### Chebyshev Polynomials

There are two types of Chebyshev polynomials: **Chebyshev polynomials of the first kind**, $ \{ T_n(x) \} $, and **Chebyshev polynomials of the second kind**, $ \{ U_n(x) \} $. We typically focus on the first kind. They are defined on the interval $[-1, 1]$ and satisfy the orthogonality condition:

$$
\int_{-1}^{1} \frac{T_n(x) T_m(x)}{\sqrt{1 - x^2}} \, dx = 
\begin{cases}
0 & \text{if } n \neq m, \\
\frac{\pi}{2} & \text{if } n = m \neq 0, \\
\pi & \text{if } n = m = 0.
\end{cases}
$$

The Chebyshev polynomials of the first kind can be generated using the recurrence relation:

$$
T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x),
$$

with initial conditions:

$$
T_0(x) = 1, \quad T_1(x) = x.
$$

Remarkably, this recurrence relation also admits an explicit formula: 

$$
T_n(x) = \cos(n \cos^{-1}(x)).
$$


Let's now implement it in Python:

```{code-cell} ipython3
:tags: [hide-input]
def chebyshev_polynomial(n, x):
    if n == 0:
        return np.poly1d([1])
    elif n == 1:
        return x
    else:
        t0 = np.poly1d([1])
        t1 = x
        for _ in range(2, n + 1):
            t2 = 2 * x * t1 - t0
            t0, t1 = t1, t2
        return t1

def chebyshev_coefficients(n):
    x = np.poly1d([1, 0])  # Define a poly1d object to represent x
    poly = chebyshev_polynomial(n, x)
    return poly

for n in range(4):
    poly = chebyshev_coefficients(n)
    display(Math(f"T_{n}(x) = {poly_to_latex(poly)}"))
```

##### Hermite Polynomials

Hermite polynomials $ \{ H_n(x) \} $ are defined on the entire real line and are orthogonal with respect to the weight function $ w(x) = e^{-x^2} $. They satisfy the orthogonality condition:

$$
\int_{-\infty}^{\infty} H_n(x) H_m(x) e^{-x^2} \, dx = 
\begin{cases}
0 & \text{if } n \neq m, \\
2^n n! \sqrt{\pi} & \text{if } n = m.
\end{cases}
$$

Hermite polynomials can be generated using the recurrence relation:

$$
H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x),
$$

with initial conditions:

$$
H_0(x) = 1, \quad H_1(x) = 2x.
$$

The following code computes the coefficients of the first four Hermite polynomials: 

```{code-cell} ipython3
:tags: [hide-input]
def hermite_polynomial(n, x):
    if n == 0:
        return np.poly1d([1])
    elif n == 1:
        return 2 * x
    else:
        h0 = np.poly1d([1])
        h1 = 2 * x
        for k in range(2, n + 1):
            h2 = 2 * x * h1 - 2 * (k - 1) * h0
            h0, h1 = h1, h2
        return h1

def hermite_coefficients(n):
    x = np.poly1d([1, 0])  # Define a poly1d object to represent x
    poly = hermite_polynomial(n, x)
    return poly

for n in range(4):
    poly = hermite_coefficients(n)
    display(Math(f"H_{n}(x) = {poly_to_latex(poly)}"))
```

### Collocation Conditions

Consider a general ODE of the form:

$$
\dot{y}(t) = f(y(t), t), \quad y(t_0) = y_0,
$$

where $ y(t) \in \mathbb{R}^n $ is the state vector, and $ f: \mathbb{R}^n \times \mathbb{R} \rightarrow \mathbb{R}^n $ is a known function. The goal is to approximate the solution $ y(t) $ over a given interval $[t_0, t_f]$. Collocation methods achieve this by:

1. **Choosing a basis** to approximate $ y(t) $ using a finite sum of basis functions $ \phi_i(t) $:

   $$
   y(t) \approx \sum_{i=0}^{N} c_i \phi_i(t),
   $$

   where $ \{c_i\} $ are the coefficients to be determined.

2. **Selecting collocation points** $ t_1, t_2, \ldots, t_N $ within the interval $[t_0, t_f]$. These are typically chosen to be the roots of certain orthogonal polynomials, like Legendre or Chebyshev polynomials, or can be spread equally across the interval.

3. **Enforcing the ODE at the collocation points** for each $ t_j $:

   $$
   \dot{y}(t_j) = f(y(t_j), t_j).
   $$

   To implement this, we differentiate the approximate solution $ y(t) $ with respect to time:

   $$
   \dot{y}(t) \approx \sum_{i=0}^{N} c_i \dot{\phi}_i(t).
   $$

   Substituting this into the ODE at the collocation points gives:

   $$
   \sum_{i=0}^{N} c_i \dot{\phi}_i(t_j) = f\left(\sum_{i=0}^{N} c_i \phi_i(t_j), t_j\right), \quad j = 1, \ldots, N.
   $$

The collocation equations are formed by enforcing the ODE at all collocation points, leading to a system of nonlinear equations:

$$
\sum_{i=0}^{N} c_i \dot{\phi}_i(t_j) - f\left(\sum_{i=0}^{N} c_i \phi_i(t_j), t_j\right) = 0, \quad j = 1, \ldots, N.
$$

Furthermore when solving an initial value problem (IVP),  we also need to incorporate the initial condition $ y(t_0) = y_0 $ as an additional constraint:

$$
\sum_{i=0}^{N} c_i \phi_i(t_0) = y_0.
$$

The collocation conditions and IVP condition are combined together to form a root-finding problem, which we can generically solve numerically using Newton's method. 

### Common Numerical Integration Techniques as Collocation Methods

Many common numerical integration techniques can be viewed as special cases of collocation methods. 
While the general collocation method we discussed earlier applies to the entire interval $[t_0, t_f]$, many numerical integration techniques can be viewed as collocation methods applied locally, step by step.

In practical numerical integration, we often divide the full interval $[t_0, t_f]$ into smaller subintervals or steps. In general, this allows us to user simpler basis functions thereby reducing computational complexity, and gives us more flexibility in dynamically ajusting the step size using local error estimates. When we apply collocation locally, we're essentially using the collocation method to "step" from $t_n$ to $t_{n+1}$. As we did, earlier we still apply the following three steps:

1. We choose a basis function to approximate $y(t)$ over $[t_n, t_{n+1}]$.
2. We select collocation points within this interval.
3. We enforce the ODE at these points to determine the coefficients of our basis function.

We can make this idea clearer by re-deriving some of the numerical integration methods seen before using this perspective. 

#### Explicit Euler Method

For the Explicit Euler method, we use a linear basis function for each step:

$$
\phi(t) = 1 + c(t - t_n)
$$

Note that we use $(t - t_n)$ rather than just $t$ because we're approximating the solution locally, relative to the start of each step. We then choose one collocation point at $t_{n+1}$ where we have:

$$
y'(t_{n+1}) = c = f(y_n, t_n)
$$

Our local approximation is:

$$
y(t) \approx y_n + c(t - t_n)
$$

At $t = t_{n+1}$, this gives:

$$
y_{n+1} = y_n + c(t_{n+1} - t_n) = y_n + hf(y_n, t_n)
$$

where $h = t_{n+1} - t_n$. This is the classic Euler update formula.

#### Implicit Euler Method

The Implicit Euler method uses the same linear basis function:

$$
\phi(t) = 1 + c(t - t_n)
$$

Again, we choose one collocation point at $t_{n+1}$. The main difference is that we enforce the ODE using $y_{n+1}$:

$$
y'(t_{n+1}) = c = f(y_{n+1}, t_{n+1})
$$

Our approximation remains:

$$
y(t) \approx y_n + c(t - t_n)
$$

At $t = t_{n+1}$, this leads to the implicit equation:

$$
y_{n+1} = y_n + hf(y_{n+1}, t_{n+1})
$$

#### Trapezoidal Method

The Trapezoidal method uses a quadratic basis function:

$$
\phi(t) = 1 + c(t - t_n) + a(t - t_n)^2
$$

We use two collocation points: $t_n$ and $t_{n+1}$. Enforcing the ODE at these points gives:

- At $t_n$:

$$
y'(t_n) = c = f(y_n, t_n)
$$

- At $t_{n+1}$:

$$
y'(t_{n+1}) = c + 2ah = f(y_n + ch + ah^2, t_{n+1})
$$

Our approximation is:

$$
y(t) \approx y_n + c(t - t_n) + a(t - t_n)^2
$$

At $t = t_{n+1}$, this gives:

$$
y_{n+1} = y_n + ch + ah^2
$$

Solving the system of equations leads to the trapezoidal update:

$$
y_{n+1} = y_n + \frac{h}{2}[f(y_n, t_n) + f(y_{n+1}, t_{n+1})]
$$

#### Runge-Kutta Methods

Higher-order Runge-Kutta methods can also be interpreted as collocation methods. The RK4 method corresponds to a collocation method using a cubic polynomial basis:

$$
\phi(t) = 1 + c_1(t - t_n) + c_2(t - t_n)^2 + c_3(t - t_n)^3
$$

Here, we're using a cubic polynomial to approximate the solution over each step, rather than the linear or quadratic approximations of the other methods above. For RK4, we use four collocation points:

1. $t_n$ (the start of the step)
2. $t_n + h/2$
3. $t_n + h/2$
4. $t_n + h$ (the end of the step)

These points are called the "Gauss-Lobatto" points, scaled to our interval $[t_n, t_n + h]$.
The RK4 method enforces the ODE at these collocation points, leading to four stages:

$$
\begin{aligned}
k_1 &= hf(y_n, t_n) \\
k_2 &= hf(y_n + \frac{1}{2}k_1, t_n + \frac{h}{2}) \\
k_3 &= hf(y_n + \frac{1}{2}k_2, t_n + \frac{h}{2}) \\
k_4 &= hf(y_n + k_3, t_n + h)
\end{aligned}
$$

The final update formula for RK4 can be derived by solving the system of equations resulting from enforcing the ODE at our collocation points:

$$
y_{n+1} = y_n + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)
$$

### Example: Solving a Simple ODE by Collocation

Consider a simple ODE:

$$
\frac{dy}{dt} = -y, \quad y(0) = 1, \quad t \in [0, 2]
$$

The analytical solution is $y(t) = e^{-t}$. We apply the collocation method with a monomial basis of order $N$:

$$
\phi_i(t) = t^i, \quad i = 0, 1, \ldots, N
$$

We select $N$ equally spaced points $\{t_1, \ldots, t_N\}$ in $[0, 2]$ as collocation points. 


```{code-cell} ipython3
:tags: [hide-input]
:load: code/collocation_ivp_demo.py
```

## Nonlinear Programming

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


### Projected Gradient Descent

The Arrow-Hurwicz-Uzawa algorithm provided a way to handle constraints through dual variables and a primal-dual update scheme. Another commonly used approach for constrained optimization is **Projected Gradient Descent (PGD)**. The idea is simple: take a gradient descent step as if the problem were unconstrained, then project the result back onto the feasible set. Formally:

$$
\mathbf{x}_{k+1} = \mathcal{P}_C\big(\mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k)\big),
$$

where \$\mathcal{P}\_C\$ is the projection onto the feasible set \$C\$, \$\alpha\$ is the step size, and \$f(\mathbf{x})\$ is the objective function.

PGD is particularly effective when the projection is computationally cheap. A common example is **box constraints** (or bound constraints), where the feasible set is a hyperrectangle:

$$
C = \{ \mathbf{x} \mid \mathbf{x}_{\mathrm{lb}} \leq \mathbf{x} \leq \mathbf{x}_{\mathrm{ub}} \}.
$$

In this case, the projection reduces to an element-wise clipping operation:

$$
[\mathcal{P}_C(\mathbf{x})]_i = \max\big(\min([\mathbf{x}]_i, [\mathbf{x}_{\mathrm{ub}}]_i), [\mathbf{x}_{\mathrm{lb}}]_i\big).
$$

For bound-constrained problems, PGD is almost as easy to implement as standard gradient descent because the projection step is just a clipping operation. For more general constraints, however, the projection may require solving a separate optimization problem, which can be as hard as the original task. Here is the algorithm for a problem of the form:

$$
\begin{aligned}
\min_{\mathbf{x}} \quad & f(\mathbf{x}) \\
\text{subject to} \quad & \mathbf{x}_{\mathrm{lb}} \leq \mathbf{x} \leq \mathbf{x}_{\mathrm{ub}}.
\end{aligned}
$$

```{prf:algorithm} Projected Gradient Descent for Bound Constraints
:label: proj-grad-descent-bound-constraints

**Input:** Initial point $\mathbf{x}_0$, step size $\alpha$, bounds $\mathbf{x}_{\mathrm{lb}}, \mathbf{x}_{\mathrm{ub}}$, 
           maximum iterations $\max_\text{iter}$, tolerance $\varepsilon$

1. Initialize $k = 0$
2. While $k < \max_\text{iter}$ and not converged:
    1. Compute gradient: $\mathbf{g}_k = \nabla f(\mathbf{x}_k)$
    2. Update: $\mathbf{x}_{k+1} = \text{clip}(\mathbf{x}_k - \alpha \mathbf{g}_k,\; \mathbf{x}_{\mathrm{lb}}, \mathbf{x}_{\mathrm{ub}})$
    3. Check convergence: if $\|\mathbf{x}_{k+1} - \mathbf{x}_k\| < \varepsilon$, stop
    4. $k = k + 1$
3. Return $\mathbf{x}_k$
```

The clipping function is defined as:

$$
\text{clip}(x, x_{\mathrm{lb}}, x_{\mathrm{ub}}) = \max\big(\min(x, x_{\mathrm{ub}}), x_{\mathrm{lb}}\big).
$$

Under mild conditions such as Lipschitz continuity of the gradient, PGD converges to a stationary point of the constrained problem. Its simplicity and low cost make it a common choice whenever the projection can be computed efficiently.

# The Discrete-Time Pontryagin Maximum Principle
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

<!-- ## Hamiltonian Formulation

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
$$ --> -->
