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
## Collocation Methods

The numerical integration methods we've discussed before are inherently sequential: given an initial state, we make a guess as to where the system might go over a small time interval. The accuracy of that guess depends on the numerical integration procedure being used and the information available locally. 

Collocation methods offer an alternative paradigm. Instead of solving the problem step by step, these methods aim to solve for the entire trajectory simultaneously by reformulating the differential equation into a system of algebraic equations. This approach involves a process of iterative refinements for the values of the discretized system, taking into account both local and global properties of the function.

Sequential methods like Euler's or Runge-Kutta focus on evolving the system forward in time, starting from known initial conditions. They are simple to use for initial value problems but can accumulate errors over long time scales. Collocation methods, on the other hand, consider the whole time domain at once and tend to distribute errors more evenly across the domain. 

This global view has some interesting implications. For one, collocation methods can handle both initial value and boundary value problems more naturally, which is particularly useful when you have path constraints throughout the entire integration interval. This property is especially valuable when solving continuous-time optimal control problems with such constraints: something which would otherwise be difficult to achieve with single-shooting methods. 

However, solving for all time points simultaneously can be computationally intensive. It involves a tradeoff between overall accuracy, especially for long-time integrations or sensitive systems, but at the cost of increased computational complexity. Despite this, the ability to handle complex constraints and achieve more robust solutions often makes collocation methods the preferred choice for solving sophisticated COCPs in fields such as aerospace, robotics, and process control.

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

In practical numerical integration, we often divide the full interval $[t_0, t_f]$ into smaller subintervals or steps. In general, this allows us to use simpler basis functions thereby reducing computational complexity, and gives us more flexibility in dynamically ajusting the step size using local error estimates. When we apply collocation locally, we're essentially using the collocation method to "step" from $t_n$ to $t_{n+1}$. As we did, earlier we still apply the following three steps:

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

## Transforming Lagrange and Bolza Problems to Mayer 

Recall that, according to the fundamental theorem of calculus, solving an initial value problem (IVP) or using a quadrature method to evaluate an integral are essentially two approaches to the same problem. Similarly, just as we can convert discrete-time Lagrange or Bolza optimal control problems to the Mayer form by keeping a running sum of the cost, we can extend this idea to continuous time. This is done by introducing an augmented state space that includes the accumulated cost up to a given time as part of the integral term. If our original system is represented by $\mathbf{f}$, we construct an augmented system $\tilde{\mathbf{f}}$ as follows:

$$
\begin{aligned}
\text{minimize} \quad & c(\mathbf{x}(t_f)) + z(t_f) \\
\text{subject to} \quad & \begin{bmatrix} \dot{\mathbf{x}}(t) \\ \dot{z}(t) \end{bmatrix} = \tilde{\mathbf{f}}(\mathbf{x}(t), z(t), \mathbf{u}(t)) \\
& \mathbf{g}(\mathbf{x}(t), \mathbf{u}(t)) \leq \mathbf{0} \\
& \mathbf{x}_{\text{min}} \leq \mathbf{x}(t) \leq \mathbf{x}_{\text{max}} \\
& \mathbf{u}_{\text{min}} \leq \mathbf{u}(t) \leq \mathbf{u}_{\text{max}} \\
\text{given} \quad & \mathbf{x}(t_0) = \mathbf{x}_0, \quad z(t_0) = 0 \enspace .
\end{aligned}
$$

Here, the newly introduced state variable $z(t)$ containts the accumulated integral cost over time. The augmented system dynamics, $\tilde{\mathbf{f}}$, are defined as:

$$
\tilde{\mathbf{f}}(\mathbf{x}, z, \mathbf{u}) = \begin{bmatrix}
\mathbf{f}(\mathbf{x}, \mathbf{u}) \\
c(\mathbf{x}, \mathbf{u})
\end{bmatrix}
$$

The integral cost is now obtained via $z(t_f)$, which is added with the terminal cost $c(\mathbf{x}(t_f))$ to recover the value of the original objective in the Bolza problem. 

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

## Direct Transcription Methods

When transitioning from discrete-time to continuous-time optimal control, several new computational challenges arise:

1. The optimization variables are now continuous functions, not just discrete sequences of values stored in an array. We seek $x(t)$ and $u(t)$, which are continuous functions of time.
2. Evaluating a candidate pair $x(t)$ and $u(t)$ requires integration. This is necessary both for computing the integral term in the objective of Lagrange or Bolza problems and for satisfying the dynamics expressed as constraints.

These challenges can be viewed as problems of representation and integration. Representation is addressed through function approximation, while integration is handled using numerical integration methods. In fields like aeronautics, aerospace, and chemical engineering, control or state functions are often represented using polynomials. Instead of searching directly over the space of all possible functions, we search over the space of parameters defining these polynomials. This approach is similar to deep learning, where we parameterize complex functions as compositions of simpler nonlinear functions and adjust their weights through gradient descent, rather than searching the function space directly. Therefore, the state and control parameterization technique developed in this chapter, using polynomials, can be extended to other function approximation methods, including neural networks.

Evaluating the integral is essentially a problem of numerical integration, which naturally pairs with the numerical integration of the dynamics. Together, these elements form the foundation of many methods collectively known as direct transcription methods. The key idea is to transform the original continuous-time optimal control problem—an infinite-dimensional optimization problem—into a finite-dimensional approximation that can be solved as a standard nonlinear programming (NLP) problem, similar to those we have encountered in discrete-time optimal control.

### Direct Single Shooting

Single shooting under a direct transcription approach amounts to expressing the equality constraints of the original problem about the ODE dynamics by a time-discretized counterpart unrolled in time. This method essentially "backpropagates" through the numerical integration code that implements the system.

#### Mayer Problem
Consider a problem in Mayer form:

$$
\begin{aligned}
    \text{minimize} \quad & c(\mathbf{x}(t_f)) \\
    \text{subject to} \quad & \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)), \quad t \in [t_0, t_f] \\
                            & \mathbf{u}_{\text{min}} \leq \mathbf{u}(t) \leq \mathbf{u}_{\text{max}}, \quad t \in [t_0, t_f] \\
    \text{given} \quad & \mathbf{x}(t_0) = \mathbf{x}_0 \enspace .
\end{aligned}
$$

Our first step is to pick a control parameterization. Traditionally this would be polynomials, but given the ease of coding these methods with automatic differentiation, it might as well be a neural network. Therefore, let's write $\mathbf{u}(t; \boldsymbol{\theta})$ to denote the parameterized controls. Second, let's eliminate the approximate dynamics as constraints through a process of "simulation" via the chosen time discretization scheme. For RK4, we would for example obtain:

$$
\begin{aligned}
    \text{minimize}_{\boldsymbol{\theta}} \quad & c(\Phi(\boldsymbol{\theta}; \mathbf{x}_0)) \\
    \text{subject to} \quad & \mathbf{u}_{\text{min}} \leq \mathbf{u}(t_i; \boldsymbol{\theta}) \leq \mathbf{u}_{\text{max}}, \quad i = 0, \ldots, N-1 \\
    \text{where} \quad & \Phi = \Psi_N \circ \Psi_{N-1} \circ \cdots \circ \Psi_1 \\
    \Psi_i(\mathbf{x}) &= \mathbf{x} + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4) \\
    k_1 &= \mathbf{f}(\mathbf{x}, \mathbf{u}(t_i; \boldsymbol{\theta})) \\
    k_2 &= \mathbf{f}(\mathbf{x} + \frac{\Delta t}{2}k_1, \mathbf{u}(t_i + \frac{\Delta t}{2}; \boldsymbol{\theta})) \\
    k_3 &= \mathbf{f}(\mathbf{x} + \frac{\Delta t}{2}k_2, \mathbf{u}(t_i + \frac{\Delta t}{2}; \boldsymbol{\theta})) \\
    k_4 &= \mathbf{f}(\mathbf{x} + \Delta t k_3, \mathbf{u}(t_i + \Delta t; \boldsymbol{\theta}))
\end{aligned}
$$

Here, $\Phi(\boldsymbol{\theta}; \mathbf{x}_0)$ represents the final state $\mathbf{x}(t_f)$ obtained by integrating the system dynamics from the initial state $\mathbf{x}_0$ using the parameterized control $\mathbf{u}(t; \boldsymbol{\theta})$. The functions $\Psi_i$ represent the numerical integration steps (in this case, using the fourth-order Runge-Kutta method) for each time interval.

This reformulation transforms the infinite-dimensional optimal control problem into a finite-dimensional nonlinear programming (NLP) problem. The decision variables are now the parameters $\boldsymbol{\theta}$ of the control function, subject to bound constraints at each discretization point. The dynamic constraints are implicitly satisfied through the integration process embedded in the objective function.
However, ensuring that bound constraints on the controls are satisfied by a choice of parameters is more challenging than in the "tabular" case, i.e., when we are not using a control parameterization. This makes it more difficult to directly project or clip the parameters to satisfy the control bounds through a simple projected gradient descent step.

#### Bolza Problem 

Let's now address the numerical challenge that comes with the evaluation of the integral term in a Lagrange or Bolza-type problem:

$$
\begin{aligned}
\text{minimize} \quad & c(\mathbf{x}(t_f)) + \int_{t_0}^{t_f} c(\mathbf{x}(t), \mathbf{u}(t)) \, dt \\
\text{subject to} \quad & \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)) \\
& \mathbf{g}(\mathbf{x}(t), \mathbf{u}(t)) \leq \mathbf{0} \\
& \mathbf{u}_{\text{min}} \leq \mathbf{u}(t) \leq \mathbf{u}_{\text{max}} \\
\text{given} \quad & \mathbf{x}(t_0) = \mathbf{x}_0 \enspace .
\end{aligned}
$$

We can first start with the usual trick by which we transform the above Bolza problem into Mayer form by creating an augmented system and solving the following COCP: 

$$
\begin{aligned}
\text{minimize} \quad & c(\mathbf{x}(t_f)) + z(t_f) \\
\text{subject to} \quad & \begin{bmatrix} \dot{\mathbf{x}}(t) \\ \dot{z}(t) \end{bmatrix} = \tilde{\mathbf{f}}(\mathbf{x}(t), z(t), \mathbf{u}(t)) \\
& \mathbf{g}(\mathbf{x}(t), \mathbf{u}(t)) \leq \mathbf{0} \\
& \mathbf{u}_{\text{min}} \leq \mathbf{u}(t) \leq \mathbf{u}_{\text{max}} \\
\text{where} \quad & \tilde{\mathbf{f}}(\mathbf{x}, z, \mathbf{u}) = \begin{bmatrix}
\mathbf{f}(\mathbf{x}, \mathbf{u}) \\
c(\mathbf{x}, \mathbf{u})
\end{bmatrix}\\
\text{given} \quad & \mathbf{x}(t_0) = \mathbf{x}_0, \quad z(t_0) = 0 \enspace .
\end{aligned}
$$

We can then proceed to transcribe this problem via single-shooting just like we did above for Mayer problems: 

$$
\begin{aligned}
\text{minimize}_{\boldsymbol{\theta}} \quad & c(\Phi(\boldsymbol{\theta}; \mathbf{x}_0)) + z_N \\
\text{subject to} \quad & \mathbf{u}_{\text{min}} \leq \mathbf{u}(t_i; \boldsymbol{\theta}) \leq \mathbf{u}_{\text{max}}, \quad i = 0, \ldots, N-1 \\
\text{where} \quad & \tilde{\Phi} = \tilde{\Psi}_N \circ \tilde{\Psi}_{N-1} \circ \cdots \circ \tilde{\Psi}_1 \\
\quad & \tilde{\Psi}_i \left( \begin{bmatrix} \mathbf{x}_{i-1} \\ z_{i-1} \end{bmatrix}, \mathbf{u}_{i-1} \right) = \begin{bmatrix} \mathbf{x}_i \\ z_i \end{bmatrix}\quad \text{for} \quad i = 1, \ldots, N \enspace ,
\end{aligned}
$$

where the $\tilde \Psi$ denote a step of the underlying numerical integration method (eg. Euler, RK4, etc.) over the augmented system $\tilde{\mathbf{f}}$.

#### Decoupled Integration
The transformation of the Bolza problem into a Mayer problem has a side effect that it couples the numerical integration scheme used for the dynamics and the one for the objective itself. It certainly simplifies our implementation, but at the cost of less flexibility in the approximation of the problem as we will see below.
We could for example directly apply a numerical quadrature method (say gaussian or simpson quadrature) and RKH4, resulting in the following transcribed (single shooted) problem:

$$
\begin{aligned}
\text{minimize}{\boldsymbol{\theta}} \quad & c(\Phi_N(\boldsymbol{\theta}; \mathbf{x}_0)) + Q(\boldsymbol{\theta}) \\
\text{subject to} \quad & \mathbf{u}_{\text{min}} \leq \mathbf{u}(t_i; \boldsymbol{\theta}) \leq \mathbf{u}_{\text{max}}, \quad i = 0, \ldots, N-1 \\
\text{where} \quad & \Phi_i = \Psi_i \circ \Psi_{i-1} \circ \cdots \circ \Psi_1 \\
\quad & Q(\boldsymbol{\theta}) = \sum_{i=0}^{N} w_i c(\Phi_i(\boldsymbol{\theta}; \mathbf{x}_0), \mathbf{u}(t_i; \boldsymbol{\theta}))
\end{aligned}
$$

Here, $\Phi_i$ represents the integration of the system dynamics using RK4 as before, while $Q(\boldsymbol{\theta})$ represents the approximation of the integral cost using a chosen quadrature method. The weights $w_i$ and the number of points $N$ depend on the chosen quadrature method. For example:

1. For the trapezoidal rule, we have N+1 equally spaced points, and the weights are:

$$
w_0 = w_N = \frac{\Delta t}{2}, \quad w_i = \Delta t \text{ for } i = 1, \ldots, N-1
$$

where $\Delta t = (t_f - t_0) / N$ is the time step.

2. For Simpson's rule (assuming $N$ is even), we have N+1 equally spaced points, and the weights are:

$$
w_0 = w_N = \frac{\Delta t}{3}, \quad w_i = \frac{4\Delta t}{3} \text{ for odd } i, \quad w_i = \frac{2\Delta t}{3} \text{ for even } i \neq 0, N
$$

3. For 2-point Gaussian quadrature, we have $2N$ points (two per subinterval), and the weights are:

$$
w_{2i-1} = w_{2i} = \frac{t_f - t_0}{2N} \text{ for } i = 1, \ldots, N
$$ 

The evaluation points $t_i$ are not equally spaced in this case, but are determined by the Gaussian quadrature formula within each subinterval.

In all these cases, the states at the quadrature points are obtained from the integration of the system dynamics using the chosen method (e.g., RK4), represented by $\Phi_i(\boldsymbol{\theta}; \mathbf{x}_0)$. However, it's important to note that this approach can lead to computational inefficiencies if we're not careful. If the quadrature points don't align with the grid points used for integrating the system dynamics, we may need to perform additional interpolations or integrations to obtain the state values at the quadrature points. This mismatch can significantly increase the computational cost, especially for high-order quadrature methods or fine time discretizations.

For instance, when using Gaussian quadrature, the evaluation points are generally not equally spaced and won't coincide with the uniform time grid typically used for RK4 integration. This means we might need to integrate the dynamics to intermediate points between our main time steps, thereby increasing the number of integration steps required. Similarly, for Simpson's rule or other higher-order methods, we might need state values at points where we haven't directly computed them during our primary integration process.

To mitigate this issue, one could align the integration grid with the quadrature points, but this might compromise the accuracy of the dynamics integration if the resulting grid is not sufficiently fine. Alternatively, one could use interpolation methods to estimate the state at quadrature points, but this introduces additional approximation errors. 


#### Example: Life-Cycle Model

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

##### Direct Single Shooting Solution
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

### Direct Multiple Shooting

While direct single shooting is conceptually simple, it can suffer from numerical instabilities, especially for long time horizons or highly nonlinear systems. This phenomenon is akin to the vanishing and exploding gradient problem in deep learning (for example when unrolling an RNN over a long sequence). 

Multiple shooting addresses these issues by breaking the time interval into multiple segments, treated individually indivual single shooting problems, and stiched back together via equality constraints. Not only this approach improves numerical stability but it also allows for easier incorporation of path constraints (which was otherwise difficult to do with single shooting). Another important benefit is that each subproblem can be solved in parallel, which can be beneficial from a computational point of view.

Without loss of generality, consider a Mayer problem:

$$
\begin{aligned}
\text{minimize} \quad & c(\mathbf{x}(t_f)) \\
\text{subject to} \quad & \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)) \\
& \mathbf{g}(\mathbf{x}(t), \mathbf{u}(t)) \leq \mathbf{0} \\
& \mathbf{u}{\text{min}} \leq \mathbf{u}(t) \leq \mathbf{u}{\text{max}} \\
\text{given} \quad & \mathbf{x}(t_0) = \mathbf{x}_0 \enspace .
\end{aligned}
$$

In multiple shooting, we divide the time interval $[t_0, t_f]$ into $M$ subintervals: $[t_0, t_1], [t_1, t_2], ..., [t_{M-1}, t_M]$, where $t_M = t_f$. For each subinterval, we introduce new optimization variables $\mathbf{s}_i$ representing the initial state for that interval. The idea is that we want to make sure that the state at the end of an interval matches the initial state of the next one.

![Heat Exchanger](_static/multiple-shooting.svg)

The multiple shooting transcription of the problem becomes:

$$
\begin{aligned}
\underset{\boldsymbol{\theta}, \mathbf{s}_1, ..., \mathbf{s}_M}{\text{minimize}}  \quad & c(\mathbf{s}_M)  \\
\text{subject to} \quad & \mathbf{s}_{i+1} = \Phi_i(\boldsymbol{\theta}; \mathbf{s}_i), \quad i = 0, ..., M-1 \\
& \mathbf{g}(\mathbf{s}_i, \mathbf{u}(t_i; \boldsymbol{\theta})) \leq \mathbf{0}, \quad i = 0, ..., M \\
& \mathbf{u}_{\text{min}} \leq \mathbf{u}(t; \boldsymbol{\theta}) \leq \mathbf{u}_{\text{max}}, \quad t \in [t_0, t_f] \\
 \text{given} \quad &\mathbf{s}_0 = \mathbf{x}_0
\end{aligned}
$$

Here $\Phi_i(\boldsymbol{\theta}; \mathbf{s}_i)$ represents the final state obtained by integrating the system dynamics from $\mathbf{s}_i$ over the interval $[t_i, t_{i+1}]$ using the parameterized control $\mathbf{u}(t; \boldsymbol{\theta})$. The equality constraints $\mathbf{s}_{i+1} = \Phi_i(\boldsymbol{\theta}; \mathbf{s}_i)$ ensure continuity of the state trajectory across the subintervals. These are often called "defect constraints" or "matching conditions".



### Direct Collocation

While the single shooting method we discussed earlier eliminates the dynamics constraints through forward integration, an alternative approach is to keep these constraints explicit and solve the problem in a simultaneous manner. This approach, known as direct collocation, is part of a broader class of simultaneous methods in optimal control.

In direct collocation, instead of integrating the system forward in time, we discretize both the state and control trajectories and introduce additional constraints to ensure that the discretized trajectories satisfy the system dynamics. This method transforms the original continuous-time optimal control problem into a large-scale nonlinear programming (NLP) problem.
Let's consider our original Bolza problem:

$$
\begin{aligned}
\text{minimize} \quad & c(\mathbf{x}(t_f)) + \int_{t_0}^{t_f} c(\mathbf{x}(t), \mathbf{u}(t)) , dt \\
\text{subject to} \quad & \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)) \\
& \mathbf{g}(\mathbf{x}(t), \mathbf{u}(t)) \leq \mathbf{0} \\
& \mathbf{u}_{\text{min}} \leq \mathbf{u}(t) \leq \mathbf{u}_{\text{max}} \\
\text{given} \quad & \mathbf{x}(t_0) = \mathbf{x}_0 \enspace .
\end{aligned}
$$

In the direct collocation approach, we discretize the time domain into $N$ intervals: $t_0 < t_1 < ... < t_N = t_f$. At each node $i$, we introduce decision variables for both the state $\mathbf{x}_i$ and control $\mathbf{u}_i$. The continuous dynamics are then approximated using collocation methods, which enforce the system equations at specific collocation points within each interval.
The resulting NLP problem takes the form:

$$
\begin{aligned}
\underset{\mathbf{x}_0,\ldots,\mathbf{x}_N,\mathbf{u}_0,\ldots,\mathbf{u}_{N-1}}{\text{minimize}}
 \quad & c(\mathbf{x}_N) + \sum_{i=0}^{N-1} w_i c(\mathbf{x}_i, \mathbf{u}_i) \\
\text{subject to} \quad & \mathbf{x}_{i+1} = \mathbf{x}_i + h_i \sum_{j=1}^{k} b_j \mathbf{f}(\mathbf{x}_i^j, \mathbf{u}_i^j), \quad i = 0,...,N-1 \\
& \mathbf{g}(\mathbf{x}_i, \mathbf{u}_i) \leq \mathbf{0}, \quad i = 0,...,N-1 \\
& \mathbf{u}_{\text{min}} \leq \mathbf{u}_i \leq \mathbf{u}_{\text{max}}, \quad i = 0,...,N-1 \\
& \mathbf{x}_0 = \mathbf{x}_0^{\text{given}}\\
\end{aligned}
$$

Here, $h_i = t_{i+1} - t_i$ is the length of the i-th interval, $\mathbf{x}_i^j$ and $\mathbf{u}_i^j$ are the state and control at the j-th collocation point within the i-th interval, and $b_j$ are the weights of the collocation method. The specific form of these equations depends on the chosen collocation scheme (e.g., Hermite-Simpson, Gauss-Lobatto, etc.).

This formulation offers several advantages:

- It provides a discretized counterpart to the original continuous-time problem. 
- It allows for path constraints on both the states and controls. 
- It often results in better numerical conditioning compared to shooting methods.
- It allows facilitates the use of adaptive mesh refinement methods (which we don't cover here).

In the next sections, we'll explore specific collocation schemes and discuss their implementation in more detail.

#### Specific Collocation Methods

In the following formulations:

- $\mathbf{x}_i$ represents the state at time $t_i$.
- $\mathbf{u}_i$ represents the control at time $t_i$.
- $h_i = t_{i+1} - t_i$ is the time step.
- $\bar{\mathbf{u}}_i$ (in RK4) represents the control at the midpoint of the interval.
- $\mathbf{x}_{i+\frac{1}{2}}$ and $\mathbf{u}_{i+\frac{1}{2}}$ (in Hermite-Simpson) represent the state and control at the midpoint of the interval.

##### Euler Direct Collocation

For the Euler method, we use linear interpolation for both state and control:

$$
\begin{aligned}
\underset{\mathbf{x}_0,\ldots,\mathbf{x}_N,\mathbf{u}_0,\ldots,\mathbf{u}_{N-1}}{\text{minimize}} \quad &c(\mathbf{x}_N) + \sum_{i=0}^{N-1} h_i c(\mathbf{x}_i, \mathbf{u}_i) \\
\text{subject to} \quad & \mathbf{x}_{i+1} = \mathbf{x}_i + h_i \mathbf{f}(\mathbf{x}_i, \mathbf{u}_i, t_i), \quad i = 0,\ldots,N-1 \\
& \mathbf{g}(\mathbf{x}_i, \mathbf{u}_i) \leq \mathbf{0}, \quad i = 0,\ldots,N-1 \\
& \mathbf{u}_{\text{min}} \leq \mathbf{u}_i \leq \mathbf{u}_{\text{max}}, \quad i = 0,\ldots,N-1 \\
& \mathbf{x}_0 = \mathbf{x}_0^{\text{given}}
\end{aligned}
$$

Note that the integral of the cost function is approximated using the rectangle rule:

$$\int_{t_0}^{t_f} c(\mathbf{x}(t), \mathbf{u}(t)) dt \approx \sum_{i=0}^{N-1} h_i c(\mathbf{x}_i, \mathbf{u}_i)$$

This approximation matches the Euler integration scheme used for the dynamics:

$$\int_{t_i}^{t_{i+1}} \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t), t) dt \approx h_i \mathbf{f}(\mathbf{x}_i, \mathbf{u}_i, t_i)$$


After solving the optimization problem, we obtain discrete values for the control inputs $\mathbf{u}_i$ at each time point $t_i$. To reconstruct the continuous control function $\mathbf{u}(t)$, we use linear interpolation between these points. For $t \in [t_i, t{i+1}]$, we can express $\mathbf{u}(t)$ as:

$$\mathbf{u}(t) = \mathbf{u}_i + \frac{\mathbf{u}_{i+1} - \mathbf{u}_i}{h_i}(t - t_i)$$

This piecewise linear function provides a continuous control signal that matches the discrete optimal values found by the optimization program.

##### Trapezoidal Direct Collocation

We use the same linear interpolation as in the Euler method, but now we enforce the dynamics at both ends of the interval:

$$
\begin{aligned}
\underset{\mathbf{x}_0,\ldots,\mathbf{x}_N,\mathbf{u}_0,\ldots,\mathbf{u}_{N}}{\text{minimize}} \quad & c(\mathbf{x}_N) + \sum_{i=0}^{N-1} \frac{h_i}{2} \left[c(\mathbf{x}_i, \mathbf{u}_i) + c(\mathbf{x}_{i+1}, \mathbf{u}_{i+1})\right] \\
\text{subject to} \quad & \mathbf{x}_{i+1} = \mathbf{x}_i + \frac{h_i}{2} \left[\mathbf{f}(\mathbf{x}_i, \mathbf{u}_i, t_i) + \mathbf{f}(\mathbf{x}_{i+1}, \mathbf{u}_{i+1}, t_{i+1})\right], \quad i = 0,\ldots,N-1 \\
& \mathbf{g}(\mathbf{x}_i, \mathbf{u}_i) \leq \mathbf{0}, \quad i = 0,\ldots,N \\
& \mathbf{u}_{\text{min}} \leq \mathbf{u}_i \leq \mathbf{u}_{\text{max}}, \quad i = 0,\ldots,N \\
& \mathbf{x}_0 = \mathbf{x}_0^{\text{given}}
\end{aligned}
$$

The integral of the cost function is approximated using the trapezoidal rule:

$$\int_{t_0}^{t_f} c(\mathbf{x}(t), \mathbf{u}(t)) dt \approx \sum_{i=0}^{N-1} \frac{h_i}{2} [c(\mathbf{x}_i, \mathbf{u}_i) + c(\mathbf{x}{i+1}, \mathbf{u}{i+1})]$$

This approximation matches the trapezoidal integration scheme used for the dynamics:

$$\int_{t_i}^{t_{i+1}} \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t), t) dt \approx \frac{h_i}{2} [\mathbf{f}(\mathbf{x}_i, \mathbf{u}_i, t_i) + \mathbf{f}(\mathbf{x}{i+1}, \mathbf{u}{i+1}, t_{i+1})]$$

Similar to the Euler method, after solving the optimization problem, we obtain discrete values for the control inputs $\mathbf{u}_i$ at each time point $t_i$. The continuous control function $\mathbf{u}(t)$ is again reconstructed using linear interpolation. For $t \in [t_i, t{i+1}]$:

$$\mathbf{u}(t) = \mathbf{u}_i + \frac{\mathbf{u}_{i+1} - \mathbf{u}_i}{h_i}(t - t_i)$$

##### Hermite-Simpson Direct Collocation

For Hermite-Simpson, we use cubic interpolation for the state and quadratic for the control. The cubic interpolation comes from the fact that we use Hermite interpolation, which uses cubic polynomials to interpolate between points while matching both function values and derivatives at the endpoints. 

$$
\begin{aligned}
\underset{\mathbf{x}_0,\ldots,\mathbf{x}_N, \mathbf{u}_0,\ldots,\mathbf{u}_N, \mathbf{x}_{\frac{1}{2}},\ldots,\mathbf{x}_{N-\frac{1}{2}}, \mathbf{u}_{\frac{1}{2}},\ldots,\mathbf{u}_{N-\frac{1}{2}}}{\text{minimize}} \quad &c(\mathbf{x}_N) + \sum_{i=0}^{N-1} \frac{h_i}{6} \left[c(\mathbf{x}_i, \mathbf{u}_i) + 4c(\mathbf{x}_{i+\frac{1}{2}}, \mathbf{u}_{i+\frac{1}{2}}) + c(\mathbf{x}_{i+1}, \mathbf{u}_{i+1})\right] \\
\text{subject to} \quad & \mathbf{x}_{i+1} = \mathbf{x}_i + \frac{h_i}{6} \left[\mathbf{f}(\mathbf{x}_i, \mathbf{u}_i, t_i) + 4\mathbf{f}(\mathbf{x}_{i+\frac{1}{2}}, \mathbf{u}_{i+\frac{1}{2}}, t_{i+\frac{1}{2}}) + \mathbf{f}(\mathbf{x}_{i+1}, \mathbf{u}_{i+1}, t_{i+1})\right], \quad i = 0,\ldots,N-1 \\
& \mathbf{x}_{i+\frac{1}{2}} = \frac{1}{2}(\mathbf{x}_i + \mathbf{x}_{i+1}) + \frac{h_i}{8} \left[\mathbf{f}(\mathbf{x}_i, \mathbf{u}_i, t_i) - \mathbf{f}(\mathbf{x}_{i+1}, \mathbf{u}_{i+1}, t_{i+1})\right], \quad i = 0,\ldots,N-1 \\
& \mathbf{g}(\mathbf{x}_i, \mathbf{u}_i) \leq \mathbf{0}, \quad i = 0,\ldots,N \\
& \mathbf{g}(\mathbf{x}_{i+\frac{1}{2}}, \mathbf{u}_{i+\frac{1}{2}}) \leq \mathbf{0}, \quad i = 0,\ldots,N-1 \\
& \mathbf{u}_{\text{min}} \leq \mathbf{u}_i \leq \mathbf{u}_{\text{max}}, \quad i = 0,\ldots,N \\
& \mathbf{u}_{\text{min}} \leq \mathbf{u}_{i+\frac{1}{2}} \leq \mathbf{u}_{\text{max}}, \quad i = 0,\ldots,N-1 \\
& \mathbf{x}_0 = \mathbf{x}_0^{\text{given}}
\end{aligned}
$$

The objective function uses Simpson's rule to approximate the integral of the cost:

$$\sum_{i=0}^{N-1} \frac{h_i}{6} \left[c(\mathbf{x}_i, \mathbf{u}_i) + 4c(\mathbf{x}_{i+\frac{1}{2}}, \mathbf{u}_{i+\frac{1}{2}}) + c(\mathbf{x}_{i+1}, \mathbf{u}_{i+1})\right]$$

This is the classic Simpson's rule formula, where the integrand is evaluated at the endpoints and midpoint of each interval. The same Simpson's rule approximation is used in the dynamics constraint:

$$\mathbf{x}_{i+1} = \mathbf{x}_i + \frac{h_i}{6} \left[\mathbf{f}(\mathbf{x}_i, \mathbf{u}_i, t_i) + 4\mathbf{f}(\mathbf{x}_{i+\frac{1}{2}}, \mathbf{u}_{i+\frac{1}{2}}, t_{i+\frac{1}{2}}) + \mathbf{f}(\mathbf{x}_{i+1}, \mathbf{u}_{i+1}, t_{i+1})\right]$$

In other words, if you were to write the dynamics constraint in its representation through the fundamental theorem of calculus and apply Simpson's rule, you would obtain the above representation, which happens to correspond to Hermite interpolation: hence the name Hermite-Simpson.

Once the optimization problem is solved, we obtain discrete values for the control inputs $\mathbf{u}_i$, $\mathbf{u}_{i+\frac{1}{2}}$, and $\mathbf{u}_{i+1}$ for each interval $[t_i, t{i+1}]$. To reconstruct the continuous control function $\mathbf{u}(t)$, we use quadratic interpolation within each interval. For $t \in [t_i, t_{i+1}]$, we can express $\mathbf{u}(t)$ as:

$$\mathbf{u}(t) = a_i + b_i(t-t_i) + c_i(t-t_i)^2$$

where the coefficients $a_i$, $b_i$, and $c_i$ are determined by solving the system:

$$
\begin{aligned}
\mathbf{u}(t_i) &= \mathbf{u}_i = a_i \\
\mathbf{u}(t_i + \frac{h_i}{2}) &= \mathbf{u}_{i+\frac{1}{2}} = a_i + b_i\frac{h_i}{2} + c_i\frac{h_i^2}{4} \\
\mathbf{u}(t_{i+1}) &= \mathbf{u}_{i+1} = a_i + b_ih_i + c_ih_i^2
\end{aligned}
$$


##### Runge-Kutta 4 Direct Collocation

For RK4, we use linear interpolation for the state and piecewise constant for the control:

$$
\begin{aligned}
\underset{\mathbf{x}_0,\ldots,\mathbf{x}_N,\mathbf{u}_0,\ldots,\mathbf{u}_{N}, \bar{\mathbf{u}}_0,\ldots,\bar{\mathbf{u}}_{N-1}}{\text{minimize}} \quad & c(\mathbf{x}_N) + \sum_{i=0}^{N-1} h_i c(\mathbf{x}_i, \mathbf{u}_i) \\
\text{subject to} \quad & \mathbf{x}_{i+1} = \mathbf{x}_i + \frac{1}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4), \quad i = 0,\ldots,N-1 \\
& \mathbf{k}_1 = h_i \mathbf{f}(\mathbf{x}_i, \mathbf{u}_i, t_i) \\
& \mathbf{k}_2 = h_i \mathbf{f}(\mathbf{x}_i + \frac{1}{2}\mathbf{k}_1, \bar{\mathbf{u}}_i, t_i + \frac{h_i}{2}) \\
& \mathbf{k}_3 = h_i \mathbf{f}(\mathbf{x}_i + \frac{1}{2}\mathbf{k}_2, \bar{\mathbf{u}}_i, t_i + \frac{h_i}{2}) \\
& \mathbf{k}_4 = h_i \mathbf{f}(\mathbf{x}_i + \mathbf{k}_3, \mathbf{u}_{i+1}, t_{i+1}) \\
& \mathbf{g}(\mathbf{x}_i, \mathbf{u}_i) \leq \mathbf{0}, \quad i = 0,\ldots,N \\
& \mathbf{u}_{\text{min}} \leq \mathbf{u}_i \leq \mathbf{u}_{\text{max}}, \quad i = 0,\ldots,N \\
& \mathbf{u}_{\text{min}} \leq \bar{\mathbf{u}}_i \leq \mathbf{u}_{\text{max}}, \quad i = 0,\ldots,N-1 \\
& \mathbf{x}_0 = \mathbf{x}_0^{\text{given}}
\end{aligned}
$$

The integral of the cost function is approximated using the rectangle rule:

$$\int_{t_0}^{t_f} c(\mathbf{x}(t), \mathbf{u}(t)) dt \approx \sum_{i=0}^{N-1} h_i c(\mathbf{x}_i, \mathbf{u}_i)$$

While this is a lower-order approximation compared to the RK4 scheme used for the dynamics, it uses the same points ($t_i$) as the start of each RK4 step. Alternatively, we could also use a Simpson's rule-like, which would be more consistent with the RK4 scheme and the piecewise constant control reconstruction, but also more expensive.

Given the discrete values for the control inputs $\mathbf{u}_i$ at each time point $t_i$, and additional midpoint controls $\bar{\mathbf{u}}_i$, we finally reconstruct control function $\mathbf{u}(t)$, using a piecewise constant approach with a switch at the midpoint. For $t \in [t_i, t{i+1}]$:

$$\mathbf{u}(t) = \begin{cases}
\mathbf{u}_i & \text{if } t_i \leq t < t_i + \frac{h_i}{2} \\
\bar{\mathbf{u}}_i & \text{if } t_i + \frac{h_i}{2} \leq t < t_{i+1}
\end{cases}$$

#### Example: Compressor Surge Problem 

Compressors are mechanical devices used to increase the pressure of a gas by reducing its volume. They are found in many industrial settings, from natural gas pipelines to jet engines. However, compressors can suffer from a dangerous phenomenon called "surge" when the gas flow through the compressor falls too much below its design capacity. This can happen under different circumstances such as: 

- In a natural gas pipeline system, when there is less customer demand (e.g., during warm weather when less heating is needed) the flow through the compressor lowers.
- In a jet engine, when the pilot reduces thrust during landing, less air flows through the engine's compressors.
- In factory, the compressor might be connected through some equipment downstream via a valve. Closing it partially restricts gas flow, similar to pinching a garden hose, and can lead to compressor surge.

As the gas flow decreases, the compressor must work harder to maintain a steady flow. If the flow becomes too low, it can lead to a "breakdown": a phenomenon similar to an airplane stalling at low speeds or high angles of attack. In a compressor, when this breakdown occurs the gas briefly flows backward instead of moving forward, which in turns can cause violent oscillations in pressure which can damage the compressor and the equipments depending on it. One way to address this problem is by installing a close-coupled valve (CCV), which is a device connected at the output of the compressor to quickly modulate the flow. Our aim is to devise a optimal control approach to ensure that the compressor does not experience a surge by operating this CCV appropriately. 

Following  {cite:p}`Gravdahl1997` and {cite}`Grancharova2012`, we model the compressor using a simplified second-order representation:

$$
\begin{aligned}
\dot{x}_1 &= B(\Psi_e(x_1) - x_2 - u) \\
\dot{x}_2 &= \frac{1}{B}(x_1 - \Phi(x_2))
\end{aligned}
$$

Here, $\mathbf{x} = [x_1, x_2]^T$ represents the state variables:

- $x_1$ is the normalized mass flow through the compressor.
- $x_2$ is the normalized pressure ratio across the compressor.

The control input $u$ denotes the normalized mass flow through a CCV.
The functions $\Psi_e(x_1)$ and $\Phi(x_2)$ represent the characteristics of the compressor and valve, respectively:

$$
\begin{aligned}
\Psi_e(x_1) &= \psi_{c0} + H\left(1 + 1.5\left(\frac{x_1}{W} - 1\right) - 0.5\left(\frac{x_1}{W} - 1\right)^3\right) \\
\Phi(x_2) &= \gamma \operatorname{sign}(x_2) \sqrt{|x_2|}
\end{aligned}
$$

The system parameters are given as $\gamma = 0.5$, $B = 1$, $H = 0.18$, $\psi_{c0} = 0.3$, and $W = 0.25$.

One possible way to pose the problem {cite}`Grancharova2012` is by penalizing for the deviations to the setpoints using a quadratic penalty in the instantenous cost function as well as in the terminal one. Furthermore, we also penalize for taking large actions (which are energy hungry, and potentially unsafe) within the integral term. The idea of penalzing for deviations throughout is natural way of posing the problem when solving it via single shooting. Another alternative which we will explore below is to set the desired setpoint as a hard terminal constraint. 

The control objective is to stabilize the system and prevent surge, formulated as a continuous-time optimal control problem (COCP) in the Bolza form:

$$
\begin{aligned}
\text{minimize} \quad & \left[ \int_0^T \alpha(\mathbf{x}(t) - \mathbf{x}^*)^T(\mathbf{x}(t) - \mathbf{x}^*) + \kappa u(t)^2 \, dt\right] + \beta(\mathbf{x}(T) - \mathbf{x}^*)^T(\mathbf{x}(T) - \mathbf{x}^*) + R v^2  \\
\text{subject to} \quad & \dot{x}_1(t) = B(\Psi_e(x_1(t)) - x_2(t) - u(t)) \\
& \dot{x}_2(t) = \frac{1}{B}(x_1(t) - \Phi(x_2(t))) \\
& u_{\text{min}} \leq u(t) \leq u_{\text{max}} \\
& -x_2(t) + 0.4 \leq v \\
& -v \leq 0 \\
& \mathbf{x}(0) = \mathbf{x}_0
\end{aligned}
$$

The parameters $\alpha$, $\beta$, $\kappa$, and $R$ are non-negative weights that allow the designer to prioritize different aspects of performance (e.g., tight setpoint tracking vs. smooth control actions). We also constraint the control input to be within $0 \leq u(t) \leq 0.3$ due to the physical limitations of the valve.

The authors in {cite}`Grancharova2012` also add a soft path constraint $x_2(t) \geq 0.4$ to ensure that we maintain a minimum pressure at all time. This is implemented as a soft constraint using slack variables. The reason that we have the term $R v^2$ in the objective is to penalizes violations of the soft constraint: we allow for deviations, but don't want to do it too much.  

In the experiment below, we choose the setpoint $\mathbf{x}^* = [0.40, 0.60]^T$ as it corresponds to to an unstable equilibrium point. If we were to run the system without applying any control, we would see that the system starts to oscillate. 

```{code-cell} ipython3
:tags: [hide-input]
:load: code/compressor_surge_single_shooting.py
```

##### Solution by Trapezoidal Collocation

Another way to pose the problem is by imposing a terminal state constraint on the system rather than through a penalty in the integral term. In the following experiment, we use a problem formulation of the form: 

$$
\begin{aligned}
\text{minimize} \quad & \left[ \int_0^T \kappa u(t)^2 \, dt\right] \\
\text{subject to} \quad & \dot{x}_1(t) = B(\Psi_e(x_1(t)) - x_2(t) - u(t)) \\
& \dot{x}_2(t) = \frac{1}{B}(x_1(t) - \Phi(x_2(t))) \\
& u_{\text{min}} \leq u(t) \leq u_{\text{max}} \\
& \mathbf{x}(0) = \mathbf{x}_0 \\
& \mathbf{x}(T) = \mathbf{x}^\star
\end{aligned}
$$

We then find a control function $u(t)$ and state trajectory $x(t)$ using the trapezoidal collocation method. 

```{code-cell} ipython3
:tags: [hide-input]
:load: code/compressor_surge_trapezoidal_collocation.py
```

You can try to vary the number of collocation points in the code an observe how the state trajectory progressively matches the ground thruth (the line denoted "integrated solution"). Note that this version of the code also lacks bound constraints on the variable $x_2$ to ensure a minimum pressure, as we did earlier. Consider this a good exercise for you to try on your own. 

## System Identification 

System identification is the term used outside of machine learning communities to describe the process of inferring unknown quantities of a parameterized dynamical system from observations—in other words, learning a "world model" from data. In the simplest case, we are given a parameterized model of the system and aim to infer the values of its parameters. For example, in the compressor surge problem, one might choose to use the simplified 2nd-order model and then take measurements of the actual compressor to make the model match as closely as possible. We could measure for example the characteristics of the compressor impeller or the close-coupled valve, and then refine our models with those values. 

For instance, to characterize the compressor impeller, we might vary the mass flow rate and measure the resulting pressure increase. We would then use this data to fit the compressor characteristic function $\Psi_e(x_1)$ in our model by estimating parameters like $\psi_{c0}$, $H$, and $W$. The process of planning these experiments is sometimes optimized through an Optimal Design of Experiment phase. In this problem, we aim to determine the most efficient way to collect data, typically by gathering fewer but higher-quality samples at a lower acquisition cost. While we don't cover this material in this course, it's worth noting its relevance to the exploration and data collection problem in machine learning. The cross-pollination of ideas between optimal design of experiments and reinforcement learning could offer valuable insights to researchers and practitioners in both fields.

### Direct Single Shooting Approach 
A straightforward approach to system identification is to collect data from the system under varying operating conditions and find model parameters that best reconstruct the observed trajectories. This can be formulated as an L2 minimization problem, similar in structure to our optimal control problems, but with the objective function now including a summation over a discrete set of observations collected at specific (and potentially irregular) time intervals.
Now, one issue we face is that the interval at which the data was collected might differ from the one used by the numerical integration procedure to simulate our model. This discrepancy would make computing the reconstruction error challenging, as the number of datapoints might differ for the two processes. There are two possible ways to address this:

By aligning the time grid used by our numerical integration procedure to match that of the data. The downside of this approach is that it may compromise the accuracy of the numerical integration, especially if the data collection intervals are irregular or too large for the system's dynamics.
By using polynomial interpolation over the numerical solution to find the missing values needed to compare with the real measurements. This approach allows us to maintain a fine, regular grid for numerical integration while still comparing against irregularly sampled data.

Mathematically, if we use this second approach with Euler integration, the problem can be expressed as:

$$
\begin{aligned}
\text{minimize}_{\boldsymbol{\theta}} \quad & \sum_{k=0}^{M-1} \|\mathbf{y}_{t_f} - \hat{\mathbf{y}}_{t_k}\|_2^2 \\
\text{subject to} \quad & \mathbf{u}_{\text{min}} \leq \mathbf{u}_i \leq \mathbf{u}_{\text{max}}, \quad i = 0, \ldots, N-1 \\
\text{where} \quad & \hat{\mathbf{y}}_{t_k} = \mathbf{h}(\mathbf{x}_{t_k}; \boldsymbol{\theta}) \\
& \mathbf{x}_{i+1} = \mathbf{x}_i + \Delta t \cdot \mathbf{f}(\mathbf{x}_i, \mathbf{u}_i; \boldsymbol{\theta}), \quad i = 0, \ldots, N-1 \\
& \mathbf{x}_{t_k} = \mathbf{x}_i + \frac{t_k - t_i}{t_{i+1} - t_i} \left(\mathbf{x}_{i+1} - \mathbf{x}_i\right), \quad t_i \leq t_k < t_{i+1} \\
\text{given} \quad & \mathbf{x}_{t_0} = \mathbf{x}_0 \\
\end{aligned}
$$

The equation for $\mathbf{x}_{t_k}$ represents linear interpolation between the numerically integrated points to obtain $\mathbf{x}(t_k; \boldsymbol{\theta})$ which we need to compare with the observation collected at $t_k$. 

#### Parameter Identification in the Compressor Surge Problem 

We consider a simple form of system identification where we will attempt to recover the value of the parameter $B$ by reconstructing trajectories of our model and compararing with a dataset of trajectories collected on the real system. 

This is just a demonstration, therefore we will pretend that the real system is the one where we set the value $B=1$ in the 2nd order simplified model. We will then vary the initial conditions of the sytem by adding a Gaussian noise perturbation to the initial conditions with mean $0$ and standard deviation $0.05$. Furthermore, we will use a"do-nothing" baseline controller which we perturb with Gaussian noise with mean $0$ and standard deviation $0.01$.  

```{code-cell} ipython3
:tags: [hide-input]
:load: code/compressor_surge_data_collection.py
```

We then use this data in a direct single shooting approach where we use RK4 for numerical integration. Note that in this demonstration, the time grids align and we don't need to do use interpolation for the state trajectories. 

```{code-cell} ipython3
:tags: [hide-input]
:load: code/compressor_surge_direct_single_shooting_rk4_paramid.py
```

### Parameterization of $f$ and Neural ODEs
In our compressor surge problem, we were provided with a physically-motivated form for the function $f$. This set of equations was likely derived by scientists with deep knowledge of the physical phenomena at play (i.e., gas compression). However, in complex systems, the underlying physics might not be well understood or too complicated to model explicitly. In such cases, we might opt for a more flexible, data-driven approach.

Instead of specifying a fixed structure for $f$, we could use a "black box" model such as a neural network to learn the dynamics directly from data. 
The optimization problem remains conceptually the same as that of parameter identification. However, we are now optimizing over the parameters of the neural network that defines $f$.

Another possibility is to blend the two approaches and use a grey-box model. In this approach, we typically use a physics-informed parameterization which we then supplement with a black-box model to account for the discrepancies in the observations. Mathematically, this can be expressed as:

$$
\dot{\mathbf{x}}(t) = f_{\text{physics}}(\mathbf{x}, t; \boldsymbol{\theta}_{\text{physics}}) + f_{\text{NN}}(\mathbf{x}, t; \boldsymbol{\theta}_{\text{NN}})
$$

where $f_{\text{physics}}$ is the physics-based model with parameters $\boldsymbol{\theta}_{\text{physics}}$, and $f_{\text{NN}}$ is a neural network with parameters $\boldsymbol{\theta}_{\text{NN}}$ that captures unmodeled dynamics.

We then learn the parameters of the black-box model in tandem with the output of the given physics-based model. You can think of the combination of these two models as a neural network of its own, with the key difference being that one subnetwork (the physics-based one) has frozen weights (non-adjustable parameters).

This approach is easy to implement using automatic differentiation techniques and allows us to leverage prior knowledge to make the data-driven modelling more sample efficient. From a learning perspective, it amounts to providing inductive biases to make learning more efficient and to generalize better. 
