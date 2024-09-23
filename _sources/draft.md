
 Consider a general ODE of the form:

$$
\dot{y}(t) = f(y(t), t), \quad y(t_0) = y_0,
$$

where $ y(t) \in \mathbb{R}^n $ is the state vector, and $ f: \mathbb{R}^n \times \mathbb{R} \rightarrow \mathbb{R}^n $ is a known function. The goal is to approximate the solution $ y(t) $ over a given interval $[t_0, t_f]$. In collocation methods, we proceed to solve problem by: 

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

This set of equations can be represented in matrix form as:

$$
A \mathbf{c} = F(\mathbf{c}),
$$

where $ A $ is the matrix of derivatives of the basis functions evaluated at the collocation points, $ \mathbf{c} $ is the vector of coefficients $ [c_0, c_1, \ldots, c_N]^T $, and $ F(\mathbf{c}) $ is the vector of the function evaluations at the collocation points. For an initial value problem (IVP), we incorporate the initial condition $ y(t_0) = y_0 $ as:

$$
\sum_{i=0}^{N} c_i \phi_i(t_0) = y_0.
$$

The collocation conditions and IVP condition are combined together to form a root-finding problem, which we can generically solve numerically using Newton's method. 

### **Example: Solving a Simple ODE**

Consider a simple ODE:

$$
\dot{y}(t) = -ky(t), \quad y(0) = y_0,
$$

with the exact solution $ y(t) = y_0 e^{-kt} $. Suppose that we choose to approximate $ y(t) $ with a polynomial in the monomial basis:

$$
y(t) \approx c_0 + c_1 t.
$$

Furthermore, we choose collocation points $ t_1 = 0.25 $ and $ t_2 = 0.75 $ on $[0, 1]$. Enforcing the ODE at these points yields the following collocation equations:

$$
c_1 = -k(c_0 + c_1 \cdot 0.25), \quad c_1 = -k(c_0 + c_1 \cdot 0.75).
$$

With the initial condition $ y(0) = y_0 $, we have $ c_0 = y_0 $. Solving the system for $ c_1 $ gives an approximation for the solution over the interval.
