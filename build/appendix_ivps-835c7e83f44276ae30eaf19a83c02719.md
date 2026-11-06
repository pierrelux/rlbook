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

An ordinary differential equation specifies a trajectory through its rate of
change. An **initial value problem** (IVP) supplies the state from which that
trajectory starts:

$$
\dot{\mathbf{x}}(t)=\mathbf{f}(t,\mathbf{x}(t)),
\qquad
\mathbf{x}(t_0)=\mathbf{x}_0.
$$

Except for special vector fields, the state at a later time is not available in
closed form. A numerical integrator advances an approximation from one mesh
point to the next. Given $t_n$, an approximate state $\mathbf{x}_n$, and a step
$h_n=t_{n+1}-t_n$, one step defines a map

$$
\mathbf{x}_{n+1}
=\Phi_{h_n}(t_n,\mathbf{x}_n)
\approx \mathbf{x}(t_{n+1}).
$$

This one-step map is the part of an ODE solver used by shooting methods. The
sections below derive five fixed-step maps: forward Euler, backward Euler, the
implicit trapezoidal rule, Heun's predictor-corrector method, and classical
fourth-order Runge--Kutta.

## The integral identity behind a step

Integrating the ODE over one interval gives the exact relation

$$
\mathbf{x}(t_{n+1})
=\mathbf{x}(t_n)
+\int_{t_n}^{t_{n+1}}
\mathbf{f}(s,\mathbf{x}(s))\,ds.
$$

Every method in this appendix replaces the unknown integral by finitely many
evaluations of $\mathbf{f}$. Their locations, weights, and stage equations
jointly determine the method's accuracy, computational cost, and stability.

Two error scales are useful. The **local truncation error** measures the error
made by one step that starts from the exact state. The **global error** measures
the accumulated error after advancing across a fixed time interval. A method of
order $p$ has global error $O(h^p)$ as a uniform step $h$ tends to zero, provided
the solution and vector field are sufficiently smooth.

## Forward (explicit) Euler

Forward Euler, also called **explicit Euler**, replaces the vector field
throughout the interval by its value at the left endpoint:

$$
\mathbf{x}_{n+1}
=\mathbf{x}_n+h_n\mathbf{f}(t_n,\mathbf{x}_n).
$$

The update is explicit because every quantity on the right is known. It uses
one vector-field evaluation per step and has first-order global accuracy. The
same formula follows from the first-order Taylor expansion of
$\mathbf{x}(t_n+h_n)$.

For the ballistic system

$$
\dot y=v,
\qquad
\dot v=-g,
$$

forward Euler gives

$$
y_{n+1}=y_n+h_nv_n,
\qquad
v_{n+1}=v_n-h_ng.
$$

The velocity update is exact because its derivative is constant. The position
update is not exact because the velocity changes during the interval.

## Backward (implicit) Euler

Backward Euler, also called **implicit Euler**, evaluates the vector field at
the unknown right endpoint:

$$
\mathbf{x}_{n+1}
=\mathbf{x}_n+h_n\mathbf{f}(t_{n+1},\mathbf{x}_{n+1}).
$$

This equation is implicit. One step requires solving the nonlinear residual

$$
\mathbf{r}(\mathbf{z})
:=\mathbf{z}-\mathbf{x}_n
-h_n\mathbf{f}(t_{n+1},\mathbf{z})
=\mathbf{0}
$$

for $\mathbf{z}=\mathbf{x}_{n+1}$. A fixed-point iteration or a root finder such
as Newton's method can perform the internal solve. A failed or loosely converged
internal solve changes the effective one-step map, so its residual tolerance is
part of the integration accuracy.

Backward Euler has first-order global accuracy. Its main advantage is stability:
strongly decaying modes remain stable for step sizes that would make forward
Euler diverge.

## Implicit trapezoidal rule

The trapezoidal rule approximates the integral of the vector field by the
average of its values at the two endpoints:

$$
\mathbf{x}_{n+1}
=\mathbf{x}_n
+\frac{h_n}{2}
\left[
\mathbf{f}(t_n,\mathbf{x}_n)
+\mathbf{f}(t_{n+1},\mathbf{x}_{n+1})
\right].
$$

The right-endpoint state is unknown, so this rule is also implicit. Its
nonlinear residual is

$$
\mathbf{r}(\mathbf{z})
:=\mathbf{z}-\mathbf{x}_n
-\frac{h_n}{2}
\left[
\mathbf{f}(t_n,\mathbf{x}_n)
+\mathbf{f}(t_{n+1},\mathbf{z})
\right].
$$

Solving $\mathbf{r}(\mathbf{z})=\mathbf{0}$ to sufficient accuracy produces a
second-order method. The implicit trapezoidal rule is time-symmetric and stable
throughout the left half of the complex plane. Unlike backward Euler, however,
it does not strongly damp extremely fast modes; a large step can make such modes
alternate in sign while decaying very slowly.

## Predictor-corrector iteration and Heun's method

An Euler prediction supplies a convenient initial guess for the implicit
trapezoidal equation:

$$
\mathbf{x}_{n+1}^{(0)}
=\mathbf{x}_n+h_n\mathbf{f}(t_n,\mathbf{x}_n).
$$

Substituting that prediction into the right-endpoint evaluation gives

$$
\mathbf{x}_{n+1}^{(1)}
=\mathbf{x}_n
+\frac{h_n}{2}
\left[
\mathbf{f}(t_n,\mathbf{x}_n)
+\mathbf{f}(t_{n+1},\mathbf{x}_{n+1}^{(0)})
\right].
$$

Stopping after this correction gives **Heun's method**, also called the
explicit trapezoidal method. It is an explicit, second-order Runge--Kutta
method: the endpoint slope is evaluated at the Euler-predicted state, not at the
unknown final state.

Repeating the correction,

$$
\mathbf{x}_{n+1}^{(m+1)}
=\mathbf{x}_n
+\frac{h_n}{2}
\left[
\mathbf{f}(t_n,\mathbf{x}_n)
+\mathbf{f}(t_{n+1},\mathbf{x}_{n+1}^{(m)})
\right],
$$

is a fixed-point, or Picard, iteration for the implicit trapezoidal equation. If
the iteration converges, its limit is the implicit trapezoidal step. It is not a
Newton iteration. Newton's method instead uses the residual Jacobian and updates

$$
\mathbf{z}^{(m+1)}
=\mathbf{z}^{(m)}
-\left[
\mathbf{I}-\frac{h_n}{2}
\nabla_{\mathbf{x}}\mathbf{f}(t_{n+1},\mathbf{z}^{(m)})
\right]^{-1}
\mathbf{r}(\mathbf{z}^{(m)}).
$$

The distinction matters when the step is large or the dynamics are stiff.
Picard iteration requires its fixed-point map to be contractive, whereas a
well-globalized Newton solve can converge in cases where Picard iteration does
not.

## Classical fourth-order Runge--Kutta

Classical RK4 evaluates four slopes. Starting from $(t_n,\mathbf{x}_n)$, define

$$
\begin{aligned}
\mathbf{k}_1 &= \mathbf{f}(t_n,\mathbf{x}_n),\\
\mathbf{k}_2 &= \mathbf{f}\!\left(t_n+\frac{h_n}{2},
  \mathbf{x}_n+\frac{h_n}{2}\mathbf{k}_1\right),\\
\mathbf{k}_3 &= \mathbf{f}\!\left(t_n+\frac{h_n}{2},
  \mathbf{x}_n+\frac{h_n}{2}\mathbf{k}_2\right),\\
\mathbf{k}_4 &= \mathbf{f}\!\left(t_n+h_n,
  \mathbf{x}_n+h_n\mathbf{k}_3\right).
\end{aligned}
$$

The step is

$$
\mathbf{x}_{n+1}
=\mathbf{x}_n
+\frac{h_n}{6}
\left(\mathbf{k}_1+2\mathbf{k}_2+2\mathbf{k}_3+\mathbf{k}_4\right).
$$

RK4 is explicit and has fourth-order global accuracy for smooth problems. The
two midpoint evaluations occur at the same time but at different predicted
states. They are stages of the Runge--Kutta calculation, not values of one
cubic interpolant at four distinct collocation nodes. Classical RK4 is therefore
not a collocation method. Some *implicit* Runge--Kutta methods are obtained from
Gauss, Radau, or Lobatto collocation, but that correspondence does not include
classical RK4.

## A compact fixed-step implementation

The implementations below expose the one-step maps directly. The implicit
methods use a generic nonlinear root solver initialized by a forward Euler
prediction.

```{code-cell} python
import numpy as np
from scipy.optimize import root


def forward_euler_step(fun, t, x, h):
    x = np.asarray(x, dtype=float)
    return x + h * fun(t, x)


def backward_euler_step(fun, t, x, h):
    x = np.asarray(x, dtype=float)
    guess = forward_euler_step(fun, t, x, h)
    residual = lambda z: z - x - h * fun(t + h, z)
    solution = root(residual, guess)
    residual_norm = np.linalg.norm(residual(solution.x), ord=np.inf)
    tolerance = 1e-10 * (1 + np.linalg.norm(x, ord=np.inf))
    if residual_norm > tolerance:
        raise RuntimeError(
            f"backward Euler residual {residual_norm:.3e}: {solution.message}"
        )
    return solution.x


def implicit_trapezoid_step(fun, t, x, h):
    x = np.asarray(x, dtype=float)
    slope_left = fun(t, x)
    guess = x + h * slope_left
    residual = lambda z: z - x - 0.5 * h * (slope_left + fun(t + h, z))
    solution = root(residual, guess)
    residual_norm = np.linalg.norm(residual(solution.x), ord=np.inf)
    tolerance = 1e-10 * (1 + np.linalg.norm(x, ord=np.inf))
    if residual_norm > tolerance:
        raise RuntimeError(
            f"trapezoidal residual {residual_norm:.3e}: {solution.message}"
        )
    return solution.x


def heun_step(fun, t, x, h):
    x = np.asarray(x, dtype=float)
    slope_left = fun(t, x)
    prediction = x + h * slope_left
    return x + 0.5 * h * (slope_left + fun(t + h, prediction))


def rk4_step(fun, t, x, h):
    x = np.asarray(x, dtype=float)
    k1 = fun(t, x)
    k2 = fun(t + 0.5 * h, x + 0.5 * h * k1)
    k3 = fun(t + 0.5 * h, x + 0.5 * h * k2)
    k4 = fun(t + h, x + h * k3)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_fixed_step(step, fun, x0, t0, tf, h):
    n_steps = int(np.ceil((tf - t0) / h))
    times = np.linspace(t0, tf, n_steps + 1)
    states = np.empty((n_steps + 1, np.size(x0)), dtype=float)
    states[0] = np.asarray(x0, dtype=float)
    for n in range(n_steps):
        step_size = times[n + 1] - times[n]
        states[n + 1] = step(fun, times[n], states[n], step_size)
    return times, states


decay = lambda t, x: -2.0 * x
exact_terminal = np.exp(-6.0)
methods = {
    "forward Euler": forward_euler_step,
    "backward Euler": backward_euler_step,
    "implicit trapezoid": implicit_trapezoid_step,
    "Heun": heun_step,
    "classical RK4": rk4_step,
}

for name, step in methods.items():
    _, states = integrate_fixed_step(step, decay, [1.0], 0.0, 3.0, 0.25)
    error = abs(states[-1, 0] - exact_terminal)
    print(f"{name:20s} terminal error: {error:.3e}")
```

## Stability and stiffness

Accuracy describes how the error shrinks as $h$ tends to zero. Stability asks
whether repeated numerical steps preserve the qualitative decay of the ODE at
the chosen step size. A high-order method can still be unstable when its step is
too large.

The scalar test equation

$$
\dot x=\lambda x,
\qquad \operatorname{Re}(\lambda)<0,
$$

has a decaying exact solution. Applying a one-step method gives
$x_{n+1}=R(z)x_n$ with $z=h\lambda$. The numerical solution does not grow where
$|R(z)|\leq 1$ and decays strictly where $|R(z)|<1$. The five methods have the
following stability functions:

| method | global order | explicit? | stability function $R(z)$ |
|---|---:|:---:|---|
| forward Euler | 1 | yes | $1+z$ |
| backward Euler | 1 | no | $1/(1-z)$ |
| implicit trapezoidal | 2 | no | $(1+z/2)/(1-z/2)$ |
| Heun | 2 | yes | $1+z+z^2/2$ |
| classical RK4 | 4 | yes | $1+z+z^2/2+z^3/6+z^4/24$ |

For real $\lambda<0$, forward Euler requires
$0<h<2/|\lambda|$. Backward Euler and the implicit trapezoidal rule are
**A-stable**: their stability regions contain the entire left half-plane.
Backward Euler is also **L-stable**, since $R(z)\to 0$ as $z\to-\infty$.
The trapezoidal rule instead has $R(z)\to-1$, so it can preserve a weak
step-to-step oscillation in modes that the exact solution damps almost
immediately. Heun and RK4 can give higher accuracy per step than forward Euler
on smooth nonstiff problems, but both remain explicit methods with bounded
stability regions.

An IVP is **stiff** when rapidly decaying modes force an explicit method to take
steps much shorter than the time scale on which the desired solution changes.
Implicit methods cost more per step because they solve nonlinear equations, but
their stability can make much larger steps possible. Stability does not remove
the need for accuracy checks: a stable coarse solution can still be inaccurate.

## The one-step map in shooting methods

A controlled system adds a prescribed input over each integration interval:

$$
\dot{\mathbf{x}}(t)
=\mathbf{f}(t,\mathbf{x}(t),\mathbf{u}(t)).
$$

For example, under a zero-order hold every stage used to advance across
$[t_n,t_{n+1}]$ receives the same value $\mathbf{u}_n$. Applying any integrator
above then defines

$$
\mathbf{x}_{n+1}
=\Phi_{h_n}(t_n,\mathbf{x}_n,\mathbf{u}_n).
$$

In **single shooting**, the optimizer selects the control parameters and every
state follows by repeatedly applying $\Phi$. The states are intermediate values
of the computation rather than independent optimization variables.

In **multiple shooting**, the initial state of each segment is an optimization
variable. Integration produces a predicted terminal state
$\widehat{\mathbf{s}}_{k+1}$ for segment $k$, and the nonlinear program imposes
the continuity defect

$$
\mathbf{s}_{k+1}-\widehat{\mathbf{s}}_{k+1}=\mathbf{0}.
$$

The integrator is therefore part of the model seen by the optimizer. Its step
size, control interpolation, nonlinear-solve tolerance, and differentiability
all affect the resulting objective and constraints. After optimization, a
tighter integration tolerance or a refined step should replay the fixed control
trajectory. A large discrepancy between the optimization rollout and this
validation rollout indicates that the integration mesh was inadequate.

## From shooting to collocation

Shooting advances a state sequentially through a one-step or segment map.
[Trajectory Optimization in Continuous Time](continuous-time-collocation.md) develops the
simultaneous alternative: state values across an interval become optimization
variables, a polynomial interpolates those values, and the ODE is imposed at
selected nodes. That chapter treats polynomial representations, differentiation
matrices, quadrature, and direct collocation. Keeping those ideas there avoids a
second, competing derivation in this IVP appendix.

## Summary

Forward Euler, Heun, and classical RK4 are explicit one-step methods of orders
one, two, and four. Backward Euler and the implicit trapezoidal rule require an
internal nonlinear solve; they have orders one and two and are stable throughout
the left half-plane. One Euler prediction followed by one trapezoidal correction
is Heun's explicit method. Repeated corrections form Picard iteration for the
implicit trapezoidal equation, while Newton's method uses the residual Jacobian.

For shooting, any of these methods becomes a finite transition map
$\Phi_h$. Single shooting chains the map from one initial state; multiple
shooting chains shorter integrations and constrains their endpoints to agree.
Direct collocation replaces that sequential construction by simultaneous nodal
variables and algebraic ODE constraints.
