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


