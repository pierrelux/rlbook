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

# Continuous-Time Trajectory Optimization 

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


## Canonical Forms

These formulations use a state-space representation, where $\mathbf{x}(t)$ denotes the state variables and $\mathbf{u}(t)$ the control inputs. The time horizon $[t_0, t_f]$ represents the initial and final times, which may be fixed or free depending on the problem. While we've specified an initial condition $\mathbf{x}(t_0) = \mathbf{x}_0$, problems may also include terminal state constraints or free terminal states. Additionally, many practical applications involve path constraints on states and controls.

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


# Example Problems

## Inverted Pendulum

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

## Heat Exchanger 

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


## Nuclear Reactor

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

## Chemotherapy

Chemotherapy is a common treatment for cancer that involves the use of drugs to kill cancer cells. However, these drugs can also have toxic effects on healthy cells in the body. To optimize the effectiveness of chemotherapy while minimizing its side effects, we can formulate an optimal control problem. Let's explore this problem in more detail.

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

## Government Corruption 

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