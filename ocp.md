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

# Mathematical Programming Approach

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

## Discrete-Time Optimal Control Problems in General

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

### Reduction to Mayer Problems

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

Let's assume that an optimal control problem has been formulated in one of the forms presented earlier and has been given to us to solve. The following section explores numerical solutions applicable to these problems, focusing on trajectory optimization. Our goal is to output an optimal control (and state trajectory) based on the given cost function and dynamics structure.

It's important to note that the methods presented here are not learning methods; they don't involve ingesting data or inferring unknown quantities from it. However, these methods represent a central component of any decision-learning system, and we will later explore how learning concepts can be incorporated.

## Constrained Optimization Approach

The mathematical programming formulation presented earlier lends itself readily to off-the-shelf solvers for nonlinear mathematical programs. For example, we can use the `scipy.optimize.minimize` function along with the SLSQP (Sequential Least Squares Programming) solver to obtain a solution to any feasible Bolza problem of the form presented below.

Before delving into the details of the numerical solver itself, let's consider an electric vehicle energy management problem. We will use this as a test bed for our algorithms, allowing us to demonstrate the practical application of these optimization techniques.
Consider an electric vehicle traversing a planned route, where we aim to optimize its energy consumption over a 20-minute journey. Our simplified model represents the vehicle's state using two variables: $x_1$, the battery state of charge as a percentage, and $x_2$, denoting the vehicle's speed in meters per second. The control input $u$, ranging from -1 to 1, represents the motor power, with negative values indicating regenerative braking and positive values representing acceleration. The problem can be formally expressed as a mathematical program in Bolza form:

$$ \begin{align*}
\min_{x, u} \quad & J = \underbrace{x_{T,1}^2 + x_{T,2}^2}_{\text{Mayer term}} + \underbrace{\sum_{t=1}^{T-1} 0.1(x_{t,1}^2 + x_{t,2}^2 + u_t^2)}_{\text{Lagrange term}} \\[2ex]
\text{subject to:} \quad & x_{t+1} = f_t(x_t, u_t), \quad t = 1, \ldots, T-1 \\[1ex]
& x_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \\[1ex]
& -1 \leq u_t \leq 1, \quad t = 1, \ldots, T-1 \\[1ex]
& -5 \leq x_{t,1}, x_{t,2} \leq 5, \quad t = 1, \ldots, T \\[2ex]
\text{where:} \quad & f_t(x_t, u_t) = \begin{bmatrix}
   x_{t,1} + 0.1x_{t,2} + 0.05u_t \\
   x_{t,2} + 0.1u_t
   \end{bmatrix} \\[1ex]
& T = 20 \\[1ex]
& x_t = \begin{bmatrix} x_{t,1} \\ x_{t,2} \end{bmatrix} \in \mathbb{R}^2, \quad u_t \in \mathbb{R}
\end{align*} $$

The system dynamics, represented by $f_t(x_t, u_t)$, describe how the battery charge and vehicle speed evolve based on the current state and control input. The initial condition $x_1 = [1, 0]^T$ indicates that the vehicle starts with a fully charged battery and zero initial speed. The constraints $-1 \leq u_t \leq 1$ and $-5 \leq x_{t,1}, x_{t,2} \leq 5$ ensure that the control inputs and state variables remain within acceptable ranges throughout the journey. This model is of course highly simplistic and neglects the nonlinear nature of battery discharge and vehicle motion due to air resistance, road grade, and vehicle mass, etc. Furthermore, our model ignores the effect of environmental factors like wind and temperature on regenerative breaking. Route-specific information such as elevation changes and speed limits are absent, as is the consideration of auxiliary power consumption such as heating and entertainment. These are all possible improvements to our models which we ignore at the moment for the sake of simplicity.

The following code demonstrate how this problem can be solved directly using `scipy.optimize.minimize` in a black box fashion. In the next sections, we will dive deeper into the mathematical underpinnings of constrained optimization and implement our own solvers. For the moment, I simply want to bring to your attention that the solution to this problem, as expressed in its original form, involves solving for two interdependent quantities: the optimal sequence of controls, and that of the states encountered when applying them to the system. Is that bug, or a feature? We'll see that it depends...

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def solve_docp(c_T, c_t, f_t, x_1, T, u_lb, u_ub, x_lb, x_ub):
    """
    Solve a Discrete-Time Optimal Control Problem of Bolza Type using scipy.minimize with SLSQP.
    
    Parameters:
    - c_T: function, terminal cost c_T(x_T)
    - c_t: function, stage cost c_t(x_t, u_t)
    - f_t: function, state transition f_t(x_t, u_t)
    - x_1: array, initial state
    - T: int, time horizon
    - u_lb, u_ub: arrays, lower and upper bounds for control inputs
    - x_lb, x_ub: arrays, lower and upper bounds for states
    
    Returns:
    - result: OptimizeResult object from scipy.optimize.minimize
    """
    
    n_x = len(x_1)
    n_u = len(u_lb)
    
    def objective(z):
        x = z[:T*n_x].reshape(T, n_x)
        u = z[T*n_x:].reshape(T, n_u)
        
        cost = c_T(x[-1])
        for t in range(T):
            cost += c_t(x[t], u[t])
        
        return cost
    
    def constraints(z):
        x = z[:T*n_x].reshape(T, n_x)
        u = z[T*n_x:].reshape(T, n_u)
        
        cons = []
        
        # State transition constraints
        for t in range(T-1):
            cons.extend(x[t+1] - f_t(x[t], u[t]))
        
        # Initial state constraint
        cons.extend(x[0] - x_1)
        
        return np.array(cons)
    
    # Set up bounds
    bounds = []
    for t in range(T):
        bounds.extend([(xl, xu) for xl, xu in zip(x_lb, x_ub)])
    for t in range(T):
        bounds.extend([(ul, uu) for ul, uu in zip(u_lb, u_ub)])
    
    # Initial guess
    z0 = np.zeros(T * (n_x + n_u))
    
    # Solve the optimization problem
    result = minimize(
        objective,
        z0,
        method='SLSQP',
        constraints={'type': 'eq', 'fun': constraints},
        bounds=bounds,
        options={'ftol': 1e-6, 'maxiter': 1000}
    )
    
    return result

def plot_results(x_opt, u_opt, T):
    """
    Plot the optimal states and control inputs.
    
    Parameters:
    - x_opt: array, optimal states
    - u_opt: array, optimal control inputs
    - T: int, time horizon
    """
    time = np.arange(T)
    
    plt.figure(figsize=(12, 8))
    
    # Plot states
    plt.subplot(2, 1, 1)
    plt.plot(time, x_opt[:, 0], label='Battery State of Charge')
    plt.plot(time, x_opt[:, 1], label='Vehicle Speed')
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.title('Optimal State Trajectories')
    plt.legend()
    plt.grid(True)
    
    # Plot control inputs
    plt.subplot(2, 1, 2)
    plt.plot(time, u_opt, label='Motor Power Input')
    plt.xlabel('Time Step')
    plt.ylabel('Control Input')
    plt.title('Optimal Control Inputs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def example_docp():
    # Define problem-specific functions and parameters
    def c_T(x_T):
        return x_T[0]**2 + x_T[1]**2
    
    def c_t(x_t, u_t):
        return 0.1 * (x_t[0]**2 + x_t[1]**2 + u_t[0]**2)
    
    def f_t(x_t, u_t):
        return np.array([
            x_t[0] + 0.1 * x_t[1] + 0.05 * u_t[0],
            x_t[1] + 0.1 * u_t[0]
        ])
    
    x_1 = np.array([1.0, 0.0])
    T = 20
    u_lb = np.array([-1.0])
    u_ub = np.array([1.0])
    x_lb = np.array([-5.0, -5.0])
    x_ub = np.array([5.0, 5.0])
    
    result = solve_docp(c_T, c_t, f_t, x_1, T, u_lb, u_ub, x_lb, x_ub)
    
    print("Optimization successful:", result.success)
    print("Optimal cost:", result.fun)
    
    # Extract optimal states and controls
    x_opt = result.x[:T*2].reshape(T, 2)
    u_opt = result.x[T*2:].reshape(T, 1)
    
    print("Optimal states:")
    print(x_opt)
    print("Optimal controls:")
    print(u_opt)
    
    # Plot the results
    plot_results(x_opt, u_opt, T)

if __name__ == "__main__":
    example_docp()
```

### Nonlinear Programming and KKT Conditions

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

In this example, the objective function $f(x_1, x_2)$ is quadratic, which can be advantageous in solution methods, especially when combined with linear constraints. The inequality constraint $g(x_1, x_2)$ defines a circular feasible region centered at $(1, 1)$ with a radius of $\sqrt{1.5}$. Additionally, the equality constraint $h(x_1, x_2)$ requires $x_2$ to lie on a sine wave function. The following code demonstrates the difference between the unconstrained, and constrained solutions to this problem. 

```{code-cell} ipython3
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the objective function
def objective(x):
    return (x[0] - 1)**2 + (x[1] - 2.5)**2

# Define the inequality constraint function
def constraint(x):
    return -(x[0] - 1)**2 - (x[1] - 1)**2 + 1.5

# Define the gradient of the objective function
def objective_gradient(x):
    return np.array([2*(x[0] - 1), 2*(x[1] - 2.5)])

# Define the gradient of the inequality constraint function
def constraint_gradient(x):
    return np.array([-2*(x[0] - 1), -2*(x[1] - 1)])

# Define the sine wave equality constraint function
def sine_wave_equality_constraint(x):
    return x[1] - (0.5 * np.sin(2 * np.pi * x[0]) + 1.5)

# Define the gradient of the sine wave equality constraint function
def sine_wave_equality_constraint_gradient(x):
    return np.array([-np.pi * np.cos(2 * np.pi * x[0]), 1])

# Define the constraints including the sine wave equality constraint
sine_wave_constraints = [{'type': 'ineq', 'fun': constraint, 'jac': constraint_gradient},  # Inequality constraint
                         {'type': 'eq', 'fun': sine_wave_equality_constraint, 'jac': sine_wave_equality_constraint_gradient}]  # Sine wave equality constraint

# Define only the inequality constraint
inequality_constraints = [{'type': 'ineq', 'fun': constraint, 'jac': constraint_gradient}]

# Initial guess
x0 = [1.25, 1.5]

# Solve the optimization problem with the sine wave equality constraint
res_sine_wave_constraint = minimize(objective, x0, method='SLSQP', jac=objective_gradient, 
                                    constraints=sine_wave_constraints, options={'disp': False})

x_opt_sine_wave_constraint = res_sine_wave_constraint.x

# Solve the optimization problem with only the inequality constraint
res_inequality_only = minimize(objective, x0, method='SLSQP', jac=objective_gradient, 
                               constraints=inequality_constraints, options={'disp': False})

x_opt_inequality_only = res_inequality_only.x

# Solve the unconstrained optimization problem for reference
res_unconstrained = minimize(objective, x0, method='SLSQP', jac=objective_gradient, options={'disp': False})
x_opt_unconstrained = res_unconstrained.x

# Generate data for visualization
x = np.linspace(-1, 4, 400)
y = np.linspace(-1, 4, 400)
X, Y = np.meshgrid(x, y)
Z = (X - 1)**2 + (Y - 2.5)**2  # Objective function values
constraint_values = (X - 1)**2 + (Y - 1)**2

# Data for sine wave constraint
x_sine = np.linspace(-1, 4, 400)
y_sine = 0.5 * np.sin(2 * np.pi * x_sine) + 1.5

# Visualization with Improved Color Scheme
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, levels=100, cmap='viridis', alpha=0.6)  # Heatmap for the objective function

# Plot all the optimal points
plt.plot(x_opt_inequality_only[0], x_opt_inequality_only[1], 'ro', label='Optimal Solution (Inequality Only)', markersize=8, markeredgecolor='black')
plt.plot(x_opt_sine_wave_constraint[0], x_opt_sine_wave_constraint[1], 'mo', label='Optimal Solution (Sine Wave Equality & Inequality)', markersize=8, markeredgecolor='black')
plt.plot(x_opt_unconstrained[0], x_opt_unconstrained[1], 'co', label='Unconstrained Minimum', markersize=8, markeredgecolor='black')

# Adjust constraint boundary colors
plt.contour(X, Y, constraint_values, levels=[1.5], colors='navy', linewidths=2, linestyles='dashed')
plt.contourf(X, Y, constraint_values, levels=[0, 1.5], colors='skyblue', alpha=0.3)

# Plot the sine wave equality constraint with a high contrast color
plt.plot(x_sine, y_sine, 'lime', linestyle='--', linewidth=2, label='Sine Wave Equality Constraint')

plt.xlim([-1, 4])
plt.ylim([-1, 4])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Example NLP')
plt.legend(loc='upper left', fontsize='small', edgecolor='black', fancybox=True)
plt.grid(True)
# Set the aspect ratio to be equal so the circle appears correctly
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
```

While this example is simple enough to convince ourselves visually of the solution to this particular problem, it falls short of providing us with actionable chracterization of what constitutes and optimal solution in general. 
The Karush-Kuhn-Tucker (KKT) conditions provide us with an answer to this problem by generalizing the first-order optimality conditions in unconstrained optimization to problems involving both equality and inequality constraints.
This result relies on the construction of an auxiliary function called the Lagrangian, defined as: 

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\mu}, \boldsymbol{\lambda})=f(\mathbf{x})+\boldsymbol{\mu}^{\top} \mathbf{g}(\mathbf{x})+\boldsymbol{\lambda}^{\top} \mathbf{h}(\mathbf{x})$$

where $\boldsymbol{\mu} \in \mathbb{R}^m$ and $\boldsymbol{\lambda} \in \mathbb{R}^\ell$ are known as Lagrange multipliers. The first-order optimality conditions then state that if $\mathbf{x}^*$, then there must exist corresponding Lagrange multipliers $\boldsymbol{\mu}^*$ and $\boldsymbol{\lambda}^*$ such that: 

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

4. Finally, for each inequality constraint, either the constraint is active (equality holds) or its corresponding Lagrange multiplier is zero at an optimal solution(**complementary slackness**)

   $$\mu_i^* g_i(\mathbf{x}^*) = 0, \quad \forall i = 1,\ldots,m$$
  
 Let's first convince ourselves of the validity of this result through an example. 



## Transformation to an Unconstrained Problem: Single Shooting Methods

Given access to unconstrained optimization solver, the easiest method to implement is by far what is known as "single shooting" in control theory. The idea of simple: rather than having to solve for the state variables as equality constraints, we transform the original equality constraint problem into an unconstrained one through "simulation", ie by recursively computing the evolution of our system for any given set of controls and initial state. In the deterministic setting, given an initial state, we can always exactly reconstruct the resulting sequence of states by "rolling out" our model, a process which some communities would refer to as "time marching". Mathematically, this amounts to forming the following unconstrained program: 

$$
\begin{align*}
\min_{\substack{\mathbf{u}_1, \ldots, \mathbf{u}_{T-1} \\ \mathbf{u}_{lb} \leq \mathbf{u}_t \leq \mathbf{u}_{lb} \, t=1,...,T-1}} \quad c_T(\boldsymbol{\phi}_{T}(\boldsymbol{u}_{1:T-1}, \boldsymbol{x}_1)) + \sum_{t=1}^{T-1} c_t(\boldsymbol{\phi}_t(\boldsymbol{u}_{1:T-1}, \boldsymbol{x}_1), \boldsymbol{u}_{t})
\end{align*}
$$

To implement this transform, we construct a set of helper functions $\boldsymbol{\phi}_1, ..., \boldsymbol{\phi}_{T-1}$ whose role is compute the state at any time $t=1, ..., T$ resulting from applying the sequence of controls starting from the initial state. We can define those functions recursively as 

$$
\begin{align*}
&\boldsymbol{\phi}_t(\boldsymbol{u}, \boldsymbol{x}_1) \triangleq \boldsymbol{f}_{t-1}(\boldsymbol{\phi}_{t-1}(\boldsymbol{u}_{1:T-1}, \boldsymbol{x}_1), \boldsymbol{u}_{t-1}), \quad t=2,...,T\\
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
import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.example_libraries import optimizers
import matplotlib.pyplot as plt

def single_shooting_ev_optimization(T=20, num_iterations=1000, step_size=0.01):
    """
    Implements the single shooting method for the electric vehicle energy optimization problem.
    
    Args:
    T: time horizon
    num_iterations: number of optimization iterations
    step_size: step size for the optimizer
    
    Returns:
    optimal_u: optimal control sequence
    """
    
    def f(x, u, t):
        return jnp.array([
            x[0] + 0.1 * x[1] + 0.05 * u,
            x[1] + 0.1 * u
        ])
    
    def c(x, u, t):
        if t == T:
            return x[0]**2 + x[1]**2
        else:
            return 0.1 * (x[0]**2 + x[1]**2 + u**2)
    
    def phi(u, x1, t):
        x = x1
        for k in range(1, t+1):
            if k > 1:
                x = f(x, u[k-2], k-1)
        return x
    
    def objective(u):
        total_cost = 0
        for t in range(1, T):
            x_t = phi(u, x1, t)
            total_cost += c(x_t, u[t-1], t)
        x_T = phi(u, x1, T)
        total_cost += c(x_T, 0.0, T)  # No control at final step
        return total_cost
    
    def clip_controls(u):
        return jnp.clip(u, -1.0, 1.0)
    
    x1 = jnp.array([1.0, 0.0])  # Initial state: full battery, zero speed
    
    # Initialize controls
    u_init = jnp.zeros(T-1)
    
    # Setup optimizer
    optimizer = optimizers.adam(step_size)
    opt_init, opt_update, get_params = optimizer
    opt_state = opt_init(u_init)
    
    @jit
    def step(i, opt_state):
        u = get_params(opt_state)
        value, grads = jax.value_and_grad(objective)(u)
        opt_state = opt_update(i, grads, opt_state)
        u = get_params(opt_state)
        u = clip_controls(u)
        opt_state = opt_init(u)
        return value, opt_state
    
    # Run optimization
    for i in range(num_iterations):
        value, opt_state = step(i, opt_state)
        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {value}")
    
    optimal_u = get_params(opt_state)
    return optimal_u

def plot_results(optimal_u, T):
    # Compute state trajectory
    x1 = jnp.array([1.0, 0.0])
    x_trajectory = [x1]
    for t in range(T-1):
        x_next = jnp.array([
            x_trajectory[-1][0] + 0.1 * x_trajectory[-1][1] + 0.05 * optimal_u[t],
            x_trajectory[-1][1] + 0.1 * optimal_u[t]
        ])
        x_trajectory.append(x_next)
    x_trajectory = jnp.array(x_trajectory)

    time = jnp.arange(T)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time, x_trajectory[:, 0], label='Battery State of Charge')
    plt.plot(time, x_trajectory[:, 1], label='Vehicle Speed')
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.title('Optimal State Trajectories')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time[:-1], optimal_u, label='Motor Power Input')
    plt.xlabel('Time Step')
    plt.ylabel('Control Input')
    plt.title('Optimal Control Inputs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Run the optimization
optimal_u = single_shooting_ev_optimization()
print("Optimal control sequence:", optimal_u)

# Plot the results
plot_results(optimal_u, T=20)
```

The approach outlined in {prf:ref}`naive-single-shooting` , and  implemented in code {ref}`naive-single-shooting-impl`, stems directly from the mathematical definition and involves recomputing the sequence of states from the begining every time that the instantenous cost function along the trajectory needs to be evaluated. This implementation has the benefit that it requires very little storage, as the only quantity that we have to maintain in addition to the running cost is the last state. However, this simplicitity and storage savings come at a steep computation cost as it requires re-computing the trajectory up to any given stage starting from the initial state. 
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

import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.example_libraries import optimizers
import matplotlib.pyplot as plt

def single_shooting_ev_optimization(T=20, num_iterations=1000, step_size=0.01):
    """
    Implements the single shooting method for the electric vehicle energy optimization problem.
    
    Args:
    T: time horizon
    num_iterations: number of optimization iterations
    step_size: step size for the optimizer
    
    Returns:
    optimal_u: optimal control sequence
    """
    
    def f(x, u, t):
        return jnp.array([
            x[0] + 0.1 * x[1] + 0.05 * u,
            x[1] + 0.1 * u
        ])
    
    def c(x, u, t):
        if t == T:
            return x[0]**2 + x[1]**2
        else:
            return 0.1 * (x[0]**2 + x[1]**2 + u**2)
    
    def compute_trajectory_and_cost(u, x1):
        x = x1
        total_cost = 0
        for t in range(1, T):
            total_cost += c(x, u[t-1], t)
            x = f(x, u[t-1], t)
        total_cost += c(x, 0.0, T)  # No control at final step
        return total_cost
    
    def objective(u):
        return compute_trajectory_and_cost(u, x1)
    
    def clip_controls(u):
        return jnp.clip(u, -1.0, 1.0)
    
    x1 = jnp.array([1.0, 0.0])  # Initial state: full battery, zero speed
    
    # Initialize controls
    u_init = jnp.zeros(T-1)
    
    # Setup optimizer
    optimizer = optimizers.adam(step_size)
    opt_init, opt_update, get_params = optimizer
    opt_state = opt_init(u_init)
    
    @jit
    def step(i, opt_state):
        u = get_params(opt_state)
        value, grads = jax.value_and_grad(objective)(u)
        opt_state = opt_update(i, grads, opt_state)
        u = get_params(opt_state)
        u = clip_controls(u)
        opt_state = opt_init(u)
        return value, opt_state
    
    # Run optimization
    for i in range(num_iterations):
        value, opt_state = step(i, opt_state)
        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {value}")
    
    optimal_u = get_params(opt_state)
    return optimal_u

def plot_results(optimal_u, T):
    # Compute state trajectory
    x1 = jnp.array([1.0, 0.0])
    x_trajectory = [x1]
    for t in range(T-1):
        x_next = jnp.array([
            x_trajectory[-1][0] + 0.1 * x_trajectory[-1][1] + 0.05 * optimal_u[t],
            x_trajectory[-1][1] + 0.1 * optimal_u[t]
        ])
        x_trajectory.append(x_next)
    x_trajectory = jnp.array(x_trajectory)

    time = jnp.arange(T)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time, x_trajectory[:, 0], label='Battery State of Charge')
    plt.plot(time, x_trajectory[:, 1], label='Vehicle Speed')
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.title('Optimal State Trajectories')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time[:-1], optimal_u, label='Motor Power Input')
    plt.xlabel('Time Step')
    plt.ylabel('Control Input')
    plt.title('Optimal Control Inputs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Run the optimization
optimal_u = single_shooting_ev_optimization()
print("Optimal control sequence:", optimal_u)

# Plot the results
plot_results(optimal_u, T=20)
```

Despite frequent mentions of automatic differentiation, it's important to note that the single shooting approaches outlined in this section need not rely on gradient-based optimization methods. In fact, one could use any method provided by `scipy.optimize.minimize`, which offers a range of options such as:

- Derivative-free methods like [Nelder-Mead Simplex](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html), suitable for problems where gradients are unavailable or difficult to compute.
- Quasi-Newton methods like [BFGS (Broyden-Fletcher-Goldfarb-Shanno)](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html), which by default uses finite differences rather than automatic differentiation to approximate gradients.

Another common strategy for single shooting methods is to use stochastic optimization techniques. For instance, random search generates a number of candidate solutions randomly and evaluates them. This approach is useful for problems with badly behaved loss landscapes or when gradient information is unreliable. More sophisticated stochastic methods include:

- Genetic Algorithms: These mimic biological evolution, using mechanisms like selection, crossover, and mutation to evolve a population of solutions over generations {cite}`holland1992genetic`. (Implemented in [DEAP](https://github.com/DEAP/deap) library)
- Simulated Annealing: Inspired by the annealing process in metallurgy, this method allows for occasional "uphill" moves to escape local minima {cite}`kirkpatrick1983optimization`. (Available in [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html))
- Particle Swarm Optimization: This technique simulates the social behavior of organisms in a swarm, with particles (candidate solutions) moving through the search space and influencing each other {cite}`kennedy1995particle`. (Implemented in [PySwarms](https://github.com/ljvmiranda921/pyswarms) library)

The selection of an optimization method for single shooting is influenced by multiple factors: problem-specific characteristics, available computational resources, and the balance between exploring the solution space and exploiting known good solutions. While gradient-based methods generally offer faster convergence when applicable, derivative-free and stochastic approaches tend to be more robust to complex non-convex loss landscapes, albeit at the cost of increased computational demands.

In practice, however, this choice is often guided by the tools at hand and the practitioners' familiarity with them. For instance, researchers with a background in deep learning tend to gravitate towards first-order gradient-based optimization techniques along with automatic differentiation for efficient derivative computation.  



```{bibliography}
```