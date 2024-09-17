# Continuous-Time Trajectory Optimization 

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

## Example Dynamical Systems

### Inverted Pendulum

The inverted pendulum is a classic problem in control theory and robotics where the goal is to stabilize a pendulum in an upright position by applying a control force at its base.
Let's consider a simplified version of the inverted pendulum, where we focus on stabilizing the pendulum's upright orientation without controlling the base's location. The state of the system can be described by the angle $\theta(t)$ and its angular velocity $\dot{\theta}(t)$. The dynamics of the system are governed by the following ordinary differential equation:

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

### Water Heater


The water heater can be modeled using a cylindrical tank with cross-section $A$. The state of the system is described by the total mass of the water $m(t)$ and its temperature $T(t)$. The control variables are the input power $P(t)$ and the inflow rate $q_\text{in}(t)$, while the disturbances are the temperature of the inflow $T_\text{in}(t)$ and the outflow rate $q_\text{out}(t)$.

The dynamics of the water heater can be described by the following system of ordinary differential equations:

\begin{align}
\dot{m}(t) &= q_\text{in}(t) - q_\text{out}(t) \\
\dot{T}(t) &= \frac{q_\text{in}(t)}{m(t)}(T_\text{in}(t) - T(t)) + \frac{1}{m(t)C}P(t)
\end{align}

where:
- $\rho$ is the density of water
- $h(t)$ is the water level in the tank
- $C$ is the specific heat capacity of water

These equations are derived from the mass balance and energy balance principles, assuming that the specific heat capacity $C$ is constant and neglecting energy losses. The first equation represents the change in total mass over time, while the second equation describes the change in temperature over time.


## Canonical Forms

These formulations use a state-space representation, where $\mathbf{x}(t)$ denotes the state variables and $\mathbf{u}(t)$ the control inputs. The time horizon $[t_0, t_f]$ represents the initial and final times, which may be fixed or free depending on the problem. While we've specified an initial condition $\mathbf{x}(t_0) = \mathbf{x}_0$, problems may also include terminal state constraints or free terminal states. Additionally, many practical applications involve path constraints on states and controls.

As studied earlier in the discrete-time setting, we consider three variants of the continuous-time optimal control problem (COCP) with path constraints and bounds:

\begin{align*}
    &\textbf{Mayer Problem:} \\
    &\text{minimize} \quad c(\mathbf{x}(t_f)) \\
    &\text{subject to} \quad \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)) \\
    &\phantom{\text{subject to}} \quad \mathbf{g}(\mathbf{x}(t), \mathbf{u}(t)) \leq \mathbf{0} \\
    &\phantom{\text{subject to}} \quad \mathbf{x}_{\text{min}} \leq \mathbf{x}(t) \leq \mathbf{x}_{\text{max}} \\
    &\phantom{\text{subject to}} \quad \mathbf{u}_{\text{min}} \leq \mathbf{u}(t) \leq \mathbf{u}_{\text{max}} \\
    &\text{given} \quad \mathbf{x}(t_0) = \mathbf{x}_0 \enspace .
\end{align*}

\begin{align*}
    &\textbf{Lagrange Problem:} \\
    &\text{minimize} \quad \int_{t_0}^{t_f} c(\mathbf{x}(t), \mathbf{u}(t)) \, dt \\
    &\text{subject to} \quad \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)) \\
    &\phantom{\text{subject to}} \quad \mathbf{g}(\mathbf{x}(t), \mathbf{u}(t)) \leq \mathbf{0} \\
    &\phantom{\text{subject to}} \quad \mathbf{x}_{\text{min}} \leq \mathbf{x}(t) \leq \mathbf{x}_{\text{max}} \\
    &\phantom{\text{subject to}} \quad \mathbf{u}_{\text{min}} \leq \mathbf{u}(t) \leq \mathbf{u}_{\text{max}} \\
    &\text{given} \quad \mathbf{x}(t_0) = \mathbf{x}_0 \enspace .
\end{align*}

\begin{align*}
    &\textbf{Bolza Problem:} \\
    &\text{minimize} \quad c(\mathbf{x}(t_f)) + \int_{t_0}^{t_f} c(\mathbf{x}(t), \mathbf{u}(t)) \, dt \\
    &\text{subject to} \quad \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)) \\
    &\phantom{\text{subject to}} \quad \mathbf{g}(\mathbf{x}(t), \mathbf{u}(t)) \leq \mathbf{0} \\
    &\phantom{\text{subject to}} \quad \mathbf{x}_{\text{min}} \leq \mathbf{x}(t) \leq \mathbf{x}_{\text{max}} \\
    &\phantom{\text{subject to}} \quad \mathbf{u}_{\text{min}} \leq \mathbf{u}(t) \leq \mathbf{u}_{\text{max}} \\
    &\text{given} \quad \mathbf{x}(t_0) = \mathbf{x}_0 \enspace .
\end{align*}

In these formulations, the additional constraints are:

- Path constraints: $\mathbf{g}(\mathbf{x}(t), \mathbf{u}(t)) \leq \mathbf{0}$, which represent constraints that must be satisfied at all times along the trajectory.
- State bounds: $\mathbf{x}_{\text{min}} \leq \mathbf{x}(t) \leq \mathbf{x}_{\text{max}}$, which specify the lower and upper bounds on the state variables.
- Control bounds: $\mathbf{u}_{\text{min}} \leq \mathbf{u}(t) \leq \mathbf{u}_{\text{max}}$, which specify the lower and upper bounds on the control inputs.

Furthermore, we may also encounter variations of the above problems under the assumption that horizon is infinite. For example: 
\begin{align*}
    &\textbf{Infinite Horizon Problem:} \\
    &\text{minimize} \quad \int_{t_0}^{\infty} e^{-\rho t} c(\mathbf{x}(t), \mathbf{u}(t)) \, dt \\
    &\text{subject to} \quad \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)) \\
    &\phantom{\text{subject to}} \quad \mathbf{g}(\mathbf{x}(t), \mathbf{u}(t)) \leq \mathbf{0} \\
    &\phantom{\text{subject to}} \quad \mathbf{x}_{\text{min}} \leq \mathbf{x}(t) \leq \mathbf{x}_{\text{max}} \\
    &\phantom{\text{subject to}} \quad \mathbf{u}_{\text{min}} \leq \mathbf{u}(t) \leq \mathbf{u}_{\text{max}} \\
    &\text{given} \quad \mathbf{x}(t_0) = \mathbf{x}_0 \enspace .
\end{align*}

## Example Problems

### Nuclear Reactor
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

### Pollution Control 

In this pollution control model, the objective is to maximize the total present value of the utility derived from food consumption while minimizing the disutility caused by DDT pollution. The model assumes that labor is the only primary factor of production, which is allocated between food production and DDT production. DDT is used as a secondary factor of production in food output but also causes pollution that can only be reduced by natural decay.

The model introduces the following notation:

- $L$: total labor force, assumed to be constant
- $v$: amount of labor used for DDT production
- $L - v$: amount of labor used for food production
- $P$: stock of DDT pollution at time $t$
- $a(v)$: rate of DDT output; $a(0) = 0$, $a'(v) > 0$, $a''(v) < 0$ for $v \geq 0$
- $\delta$: natural exponential decay rate of DDT pollution
- $C(v) = f[L - v, a(v)]$: rate of food output to be consumed; $C(v)$ is concave, $C(0) > 0$, $C(L) = 0$; $C(v)$ attains a unique maximum at $v = V > 0$ (see Fig. 11.4)
- $u(C)$: utility function of consuming the food output $C \geq 0$; $u'(0) = \infty$, $u'(C) > 0$, $u''(C) < 0$
- $h(P)$: disutility function of pollution stock $P \geq 0$; $h'(0) = 0$, $h'(P) > 0$, $h''(P) > 0$

The optimal control problem is to maximize the objective functional:

\begin{equation*}
J = \int_0^{\infty} e^{-\rho t} [u(C(v)) - h(P)] \, \mathrm{d}t
\end{equation*}

subject to the dynamics of DDT pollution:

\begin{align*}
\dot{P}(t) &= a(v) - \delta P(t), \quad P(0) = P_0 \\
v &\geq 0
\end{align*}

The state variable is the stock of DDT pollution $P(t)$, and the control variable is the amount of labor allocated to DDT production $v$. The objective functional represents the total present value of the utility of food consumption minus the disutility of pollution, discounted at a rate $\rho$.

The dynamics of DDT pollution are governed by the rate of DDT output $a(v)$ and the natural exponential decay rate $\delta$. The initial condition for the pollution stock is given by $P(0) = P_0$.

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