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

# Solving Initial Value Problems

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

```{code-cell} python
:tags: [hide-input]

#| label: appendix_ivps-cell-01

%config InlineBackend.figure_format = 'retina'
import numpy as np
import matplotlib.pyplot as plt

def f(y, t):
    """
    Derivative function for vertical motion under gravity.
    y[0] is position, y[1] is velocity.
    """
    g = 9.81  # acceleration due to gravity (m/s^2)
    return np.array([y[1], -g])

def euler_method(f, y0, t0, t_end, h):
    """
    Implement Euler's method for the entire time range.
    """
    t = np.arange(t0, t_end + h, h)
    y = np.zeros((len(t), 2))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + h * f(y[i-1], t[i-1])
    return t, y

def true_solution(t):
    """
    Analytical solution for the ballistic trajectory.
    """
    y0, v0 = 0, 20  # initial height and velocity
    g = 9.81
    return y0 + v0*t - 0.5*g*t**2, v0 - g*t

# Set up the problem
t0, t_end = 0, 4
y0 = np.array([0, 20])  # initial height = 0, initial velocity = 20 m/s

# Different step sizes
step_sizes = [1.0, 0.5, 0.1]
colors = ['r', 'g', 'b']
markers = ['o', 's', '^']

# True solution
t_fine = np.linspace(t0, t_end, 1000)
y_true, v_true = true_solution(t_fine)

# Plotting
plt.figure(figsize=(12, 8))

# Plot Euler approximations
for h, color, marker in zip(step_sizes, colors, markers):
    t, y = euler_method(f, y0, t0, t_end, h)
    plt.plot(t, y[:, 0], color=color, marker=marker, linestyle='--', 
             label=f'Euler h = {h}', markersize=6, markerfacecolor='none')

# Plot true solution last so it's on top
plt.plot(t_fine, y_true, 'k-', label='True trajectory', linewidth=2, zorder=10)

plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Height (m)', fontsize=12)
plt.title("Euler's Method: Effect of Step Size on Ballistic Trajectory Approximation", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)

# Add text to explain the effect of step size
plt.text(2.5, 15, "Smaller step sizes\nyield better approximations", 
         bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
         fontsize=10, ha='center', va='center')

plt.tight_layout()
plt.show()
```

:::{figure} #appendix_ivps-cell-01
Rendered output from the preceding code cell.
:::

Another way to understand Euler's method is through the fundamental theorem of calculus:

$$
x(t + h) = x(t) + \int_t^{t+h} f(x(\tau), \tau) d\tau
$$

We then approximate the integral term with a box of width $h$ and height $f$, and therefore of area $h f$.
```{code-cell} python
:tags: [hide-input]

#| label: appendix_ivps-cell-02

%config InlineBackend.figure_format = 'retina'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def v(t):
    """
    Velocity function for the ballistic trajectory.
    """
    v0 = 20   # initial velocity (m/s)
    g = 9.81  # acceleration due to gravity (m/s^2)
    return v0 - g * t

def position(t):
    """
    Position function (integral of velocity).
    """
    v0 = 20
    g = 9.81
    return v0*t - 0.5*g*t**2

# Set up the problem
t0, t_end = 0, 2
num_points = 1000
t = np.linspace(t0, t_end, num_points)

# Calculate true velocity and position
v_true = v(t)
x_true = position(t)

# Euler's method with a large step size for visualization
h = 0.5
t_euler = np.arange(t0, t_end + h, h)
x_euler = np.zeros_like(t_euler)

for i in range(1, len(t_euler)):
    x_euler[i] = x_euler[i-1] + h * v(t_euler[i-1])

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

# Plot velocity function and its approximation
ax1.plot(t, v_true, 'b-', label='True velocity')
ax1.fill_between(t, 0, v_true, alpha=0.3, label='True area (displacement)')

# Add rectangles with hashed pattern, ruler-like annotations, and area values
for i in range(len(t_euler) - 1):
    t_i = t_euler[i]
    v_i = v(t_i)
    rect = Rectangle((t_i, 0), h, v_i, 
                     fill=True, facecolor='red', edgecolor='r', 
                     alpha=0.15, hatch='///')
    ax1.add_patch(rect)
    
    # Add ruler-like annotations
    # Vertical ruler (height)
    ax1.annotate('', xy=(t_i, 0), xytext=(t_i, v_i),
                 arrowprops=dict(arrowstyle='<->', color='red'))
    ax1.text(t_i - 0.05, v_i/2, f'v(t{i}) = {v_i:.2f}', rotation=90, 
             va='center', ha='right', color='red', fontweight='bold')
    
    # Horizontal ruler (width)
    ax1.annotate('', xy=(t_i, -1), xytext=(t_i + h, -1),
                 arrowprops=dict(arrowstyle='<->', color='red'))
    ax1.text(t_i + h/2, -2, f'h = {h}', ha='center', va='top', 
             color='red', fontweight='bold')
    
    # Add area value in the middle of each rectangle
    area = h * v_i
    ax1.text(t_i + h/2, v_i/2, f'Area = {area:.2f}', ha='center', va='center', 
             color='white', fontweight='bold', bbox=dict(facecolor='red', edgecolor='none', alpha=0.7))

# Plot only the points for Euler's method
ax1.plot(t_euler, v(t_euler), 'ro', markersize=6, label="Euler's points")
ax1.set_ylabel('Velocity (m/s)', fontsize=12)
ax1.set_title("Velocity Function and Euler's Approximation", fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.set_ylim(bottom=-3)  # Extend y-axis to show horizontal rulers

# Plot position function and its approximation
ax2.plot(t, x_true, 'b-', label='True position')
ax2.plot(t_euler, x_euler, 'ro--', label="Euler's approximation", markersize=6, linewidth=2)

# Add vertical arrows and horizontal lines to show displacement and time step
for i in range(1, len(t_euler)):
    t_i = t_euler[i]
    x_prev = x_euler[i-1]
    x_curr = x_euler[i]
    
    # Vertical line for displacement
    ax2.plot([t_i, t_i], [x_prev, x_curr], 'g:', linewidth=2)
    
    # Horizontal line for time step
    ax2.plot([t_i - h, t_i], [x_prev, x_prev], 'g:', linewidth=2)
    
    # Add text to show the displacement value
    displacement = x_curr - x_prev
    ax2.text(t_i + 0.05, (x_prev + x_curr)/2, f'+{displacement:.2f}', 
             color='green', fontweight='bold', va='center')
    
    # Add text to show the time step
    ax2.text(t_i - h/2, x_prev - 0.5, f'h = {h}', 
             color='green', fontweight='bold', ha='center', va='top')

ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('Position (m)', fontsize=12)
ax2.set_title("Position: True vs Euler's Approximation", fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle=':', alpha=0.7)

# Add explanatory text
ax1.text(1.845, 15, "Red hashed areas show\nEuler's approximation\nof the area under the curve", 
         bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
         fontsize=10, ha='center', va='center')

plt.tight_layout()
plt.show()
```

:::{figure} #appendix_ivps-cell-02
Rendered output from the preceding code cell.
:::


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

The main difference in the Implicit Euler method is step 4, where we need to solve a (potentially nonlinear) equation to find $x_{new}$. This is typically done using iterative methods such as fixed-point iteration or Newton's method.

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

```{code-cell} python
:tags: [hide-input]

#| label: appendix_ivps-cell-03

%config InlineBackend.figure_format = 'retina'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

def v(t):
    """Velocity function for the ballistic trajectory."""
    v0 = 20   # initial velocity (m/s)
    g = 9.81  # acceleration due to gravity (m/s^2)
    return v0 - g * t

def position(t):
    """Position function (integral of velocity)."""
    v0 = 20
    g = 9.81
    return v0*t - 0.5*g*t**2

# Set up the problem
t0, t_end = 0, 2
num_points = 1000
t = np.linspace(t0, t_end, num_points)

# Calculate true velocity and position
v_true = v(t)
x_true = position(t)

# Euler's method and Trapezoid method with a large step size for visualization
h = 0.5
t_numeric = np.arange(t0, t_end + h, h)
x_euler = np.zeros_like(t_numeric)
x_trapezoid = np.zeros_like(t_numeric)

for i in range(1, len(t_numeric)):
    # Euler's method
    x_euler[i] = x_euler[i-1] + h * v(t_numeric[i-1])
    
    # Trapezoid method (implicit, so we use a simple fixed-point iteration)
    x_trapezoid[i] = x_trapezoid[i-1]
    for _ in range(5):  # 5 iterations should be enough for this simple problem
        x_trapezoid[i] = x_trapezoid[i-1] + h/2 * (v(t_numeric[i-1]) + v(t_numeric[i]))

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), sharex=True)

# Plot velocity function and its approximations
ax1.plot(t, v_true, 'b-', label='True velocity')
ax1.fill_between(t, 0, v_true, alpha=0.3, label='True area (displacement)')

# Add trapezoids and rectangles
for i in range(len(t_numeric) - 1):
    t_i, t_next = t_numeric[i], t_numeric[i+1]
    v_i, v_next = v(t_i), v(t_next)
    
    # Euler's rectangle (hashed pattern)
    rect = Rectangle((t_i, 0), h, v_i, fill=True, facecolor='red', edgecolor='r', alpha=0.15, hatch='///')
    ax1.add_patch(rect)
    
    # Trapezoid (dot pattern)
    trapezoid = Polygon([(t_i, 0), (t_i, v_i), (t_next, v_next), (t_next, 0)], 
                        fill=True, facecolor='green', edgecolor='g', alpha=0.15, hatch='....')
    ax1.add_patch(trapezoid)
    
    # Add area values
    euler_area = h * v_i
    trapezoid_area = h * (v_i + v_next) / 2
    ax1.text(t_i + h/2, v_i/2, f'Euler: {euler_area:.2f}', ha='center', va='bottom', color='red', fontweight='bold')
    ax1.text(t_i + h/2, (v_i + v_next)/4, f'Trapezoid: {trapezoid_area:.2f}', ha='center', va='top', color='green', fontweight='bold')

# Plot points for Euler's and Trapezoid methods
ax1.plot(t_numeric, v(t_numeric), 'ro', markersize=6, label="Euler's points")
ax1.plot(t_numeric, v(t_numeric), 'go', markersize=6, label="Trapezoid points")

ax1.set_ylabel('Velocity (m/s)', fontsize=12)
ax1.set_title("Velocity Function: True vs Numerical Approximations", fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle=':', alpha=0.7)

# Plot position function and its approximations
ax2.plot(t, x_true, 'b-', label='True position')
ax2.plot(t_numeric, x_euler, 'ro--', label="Euler's approximation", markersize=6, linewidth=2)
ax2.plot(t_numeric, x_trapezoid, 'go--', label="Trapezoid approximation", markersize=6, linewidth=2)

ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('Position (m)', fontsize=12)
ax2.set_title("Position: True vs Numerical Approximations", fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle=':', alpha=0.7)

# Add explanatory text
ax1.text(1.76, 17, "Red hashed areas: Euler's approximation\nGreen dotted areas: Trapezoid approximation", 
         bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
         fontsize=10, ha='center', va='center')

plt.tight_layout()
plt.show()
```

:::{figure} #appendix_ivps-cell-03
Rendered output from the preceding code cell.
:::

Algorithmically, the trapezoid method can be described as follows:

````{prf:algorithm} Trapezoid Method

**Input:** $ f(x, t) $, $ x_0 $, $ t_0 $, $ t_{end} $, $ h $

**Output:** Approximate solution $ x(t) $ at discrete time points

1. Initialize $ t = t_0 $, $ x = x_0 $
2. While $ t < t_{end} $:
3.     Set $ t_{new} = t + h $
4.     Solve for $ x_{new} $in the equation: $ x_{new} = x + \frac{h}{2}[f(x, t) + f(x_{new}, t_{new})] $
5.     Update $ t = t_{new} $
6.     Update $ x = x_{new} $
7.     Store or output the pair $ (t, x) $
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

This gives us the update rule for the trapezoid method. Recall that the forward Euler method approximates the solution by extrapolating linearly using the slope at the beginning of the interval $[t_n, t_{n+1}] $. In contrast, the backward Euler method extrapolates linearly using the slope at the end of the interval. The trapezoid method, on the other hand, averages these two slopes. This averaging provides better approximation properties than either of the methods alone, offering both stability and accuracy. Note finally that unlike the forward or backward Euler methods, the trapezoid method is also symmetric in time. This means that if you were to reverse time and apply the method backward, you would get the same results (up to numerical precision). 

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

```{code-cell} python
:tags: [hide-input]

#| label: appendix_ivps-cell-04

%config InlineBackend.figure_format = 'retina'
import numpy as np
import matplotlib.pyplot as plt

def f(y, t):
    """
    Derivative function for vertical motion under gravity.
    y[0] is position, y[1] is velocity.
    """
    g = 9.81  # acceleration due to gravity (m/s^2)
    return np.array([y[1], -g])

def true_solution(t):
    """
    Analytical solution for the ballistic trajectory.
    """
    y0, v0 = 0, 20  # initial height and velocity
    g = 9.81
    return y0 + v0*t - 0.5*g*t**2, v0 - g*t

def trapezoid_method_visual(f, y0, t0, t_end, h):
    """
    Implement the trapezoid method for the entire time range.
    Returns predictor and corrector steps for visualization.
    """
    t = np.arange(t0, t_end + h, h)
    y = np.zeros((len(t), 2))
    y_predictor = np.zeros((len(t), 2))
    y[0] = y_predictor[0] = y0
    for i in range(1, len(t)):
        # Predictor step (Euler forward)
        slope_start = f(y[i-1], t[i-1])
        y_predictor[i] = y[i-1] + h * slope_start
        
        # Corrector step
        slope_end = f(y_predictor[i], t[i])
        y[i] = y[i-1] + h * 0.5 * (slope_start + slope_end)
    
    return t, y, y_predictor

# Set up the problem
t0, t_end = 0, 2
y0 = np.array([0, 20])  # initial height = 0, initial velocity = 20 m/s
h = 0.5  # Step size

# Compute trapezoid method steps
t, y_corrector, y_predictor = trapezoid_method_visual(f, y0, t0, t_end, h)

# Plotting
plt.figure(figsize=(12, 8))

# Plot the true solution for comparison
t_fine = np.linspace(t0, t_end, 1000)
y_true, v_true = true_solution(t_fine)
plt.plot(t_fine, y_true, 'k-', label='True trajectory', linewidth=1.5)

# Plot the predictor and corrector steps
for i in range(len(t)-1):
    # Points for the predictor step
    p0 = [t[i], y_corrector[i, 0]]
    p1_predictor = [t[i+1], y_predictor[i+1, 0]]
    
    # Points for the corrector step
    p1_corrector = [t[i+1], y_corrector[i+1, 0]]
    
    # Plot predictor step
    plt.plot([p0[0], p1_predictor[0]], [p0[1], p1_predictor[1]], 'r--', linewidth=2)
    plt.plot(p1_predictor[0], p1_predictor[1], 'ro', markersize=8)
    
    # Plot corrector step
    plt.plot([p0[0], p1_corrector[0]], [p0[1], p1_corrector[1]], 'g--', linewidth=2)
    plt.plot(p1_corrector[0], p1_corrector[1], 'go', markersize=8)
    
    # Add arrows to show the predictor and corrector adjustments
    plt.arrow(p0[0], p0[1], h, y_predictor[i+1, 0] - p0[1], color='r', width=0.005, 
              head_width=0.02, head_length=0.02, length_includes_head=True, zorder=5)
    plt.arrow(p1_predictor[0], p1_predictor[1], 0, y_corrector[i+1, 0] - y_predictor[i+1, 0], 
              color='g', width=0.005, head_width=0.02, head_length=0.02, length_includes_head=True, zorder=5)

# Add legend entries for predictor and corrector steps
plt.plot([], [], 'r--', label='Predictor step (Forward Euler)')
plt.plot([], [], 'g-', label='Corrector step (Trapezoid)')

# Labels and title
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Height (m)', fontsize=12)
plt.title("Trapezoid Method: Predictor-Corrector Structure", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()
```

:::{figure} #appendix_ivps-cell-04
Rendered output from the preceding code cell.
:::

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

An **orthogonal polynomial basis** is a set of polynomials that are orthogonal to each other and form a complete basis for a certain space of functions. This means that any function within that space can be represented as a linear combination of these polynomials. 

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

```{code-cell} python
:tags: [hide-input]
#| label: appendix_ivps-cell-05

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

:::{figure} #appendix_ivps-cell-05
Rendered output from the preceding code cell.
:::

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

This recurrence relation also admits an explicit formula: 

$$
T_n(x) = \cos(n \cos^{-1}(x)).
$$


Let's now implement it in Python:

```{code-cell} python
:tags: [hide-input]
#| label: appendix_ivps-cell-06

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

:::{figure} #appendix_ivps-cell-06
Rendered output from the preceding code cell.
:::

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

```{code-cell} python
:tags: [hide-input]
#| label: appendix_ivps-cell-07

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

:::{figure} #appendix_ivps-cell-07
Rendered output from the preceding code cell.
:::

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

```{code-cell} python
:tags: [hide-input]

#| label: appendix_ivps-cell-08


import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

def ode_function(y, t):
    """Define the ODE: dy/dt = -y"""
    return -y

def solve_ode_collocation(ode_func, t_span, y0, order):
    t0, tf = t_span
    n_points = order + 1  # number of collocation points
    t_points = np.linspace(t0, tf, n_points)
    
    def collocation_residuals(coeffs):
        residuals = []
        # Initial condition residual
        y_init = sum(c * t_points[0]**i for i, c in enumerate(coeffs))
        residuals.append(y_init - y0)
        # Collocation point residuals
        for t in t_points[1:]:  # Skip the first point as it's used for initial condition
            y = sum(c * t**i for i, c in enumerate(coeffs))
            dy_dt = sum(c * i * t**(i-1) for i, c in enumerate(coeffs) if i > 0)
            residuals.append(dy_dt - ode_func(y, t))
        return residuals

    # Initial guess for coefficients
    initial_coeffs = [y0] + [0] * order

    # Solve the system of equations with more robust settings
    solution = root(collocation_residuals, initial_coeffs, 
                   method='hybr', options={'maxfev': 10000, 'xtol': 1e-8})
    
    if not solution.success:
        # Try with a different method
        solution = root(collocation_residuals, initial_coeffs, 
                       method='lm', options={'maxiter': 5000})
        
    if not solution.success:
        print(f"Warning: Collocation solver did not fully converge for order {order}")
        # Continue anyway with the best solution found

    coeffs = solution.x

    # Generate solution
    t_fine = np.linspace(t0, tf, 100)
    y_solution = sum(c * t_fine**i for i, c in enumerate(coeffs))

    return t_fine, y_solution, t_points, coeffs

# Example usage
t_span = (0, 2)
y0 = 1
orders = [1, 2, 3, 4, 5]  # Different polynomial orders to try

plt.figure(figsize=(12, 8))

for order in orders:
    t, y, t_collocation, coeffs = solve_ode_collocation(ode_function, t_span, y0, order)
    
    # Calculate y values at collocation points
    y_collocation = sum(c * t_collocation**i for i, c in enumerate(coeffs))
    
    # Plot the results
    plt.plot(t, y, label=f'Order {order}')
    plt.scatter(t_collocation, y_collocation, s=50, zorder=5)

# Plot the analytical solution
t_analytical = np.linspace(t_span[0], t_span[1], 100)
y_analytical = y0 * np.exp(-t_analytical)
plt.plot(t_analytical, y_analytical, 'k--', label='Analytical')

plt.xlabel('t')
plt.ylabel('y')
plt.title('ODE Solutions: dy/dt = -y, y(0) = 1')
plt.legend()
plt.grid(True)
plt.show()

# Print error for each order
print("Maximum absolute errors:")
for order in orders:
    t, y, _, _ = solve_ode_collocation(ode_function, t_span, y0, order)
    y_true = y0 * np.exp(-t)
    max_error = np.max(np.abs(y - y_true))
    print(f"Order {order}: {max_error:.6f}")
```

:::{figure} #appendix_ivps-cell-08
Rendered output from the preceding code cell.
:::

