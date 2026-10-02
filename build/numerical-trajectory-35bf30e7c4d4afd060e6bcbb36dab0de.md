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
# Numerical Trajectory Optimization

The Pontryagin recursion exposes the temporal structure of first-order
optimality, but a solver still needs a finite vector of variables and
constraints. Should the states remain explicit, be eliminated by simulation,
or appear only at selected segment boundaries?

Each choice recasts the discrete-time optimal control problem as a standard
nonlinear program (NLP). Collect all decision variables (states, controls, and
any auxiliary variables) into a single vector $\mathbf{z}\in\mathbb{R}^{n_z}$
and write

$$
\begin{aligned}
\min_{\mathbf{z}\in\mathbb{R}^{n_z}} \quad & F(\mathbf{z}) \\
\text{s.t.} \quad & G(\mathbf{z}) = 0, \\
& H(\mathbf{z}) \ge 0,
\end{aligned}
$$

with maps $F:\mathbb{R}^{n_z}\to\mathbb{R}$, $G:\mathbb{R}^{n_z}\to\mathbb{R}^{r_e}$, and $H:\mathbb{R}^{n_z}\to\mathbb{R}^{r_h}$. In optimal control, $G$ typically encodes dynamics and boundary conditions, while $H$ captures path and box constraints. 

There are multiple ways to arrive at (and benefit from) this NLP:

* Simultaneous (direct transcription / full discretization): keep all states and controls as variables and impose the dynamics as equality constraints. This is straightforward and exposes sparsity, but the problem can be large unless solver-side techniques (e.g., condensing) are exploited.
* Sequential (recursive elimination / single shooting): eliminate states by forward propagation from the initial condition, leaving controls as the main decision variables. This reduces dimension and constraints, but can be sensitive to initialization and longer horizons.
* Multiple shooting: introduce state variables at segment boundaries and enforce continuity between simulated segments. This compromises between size and conditioning and is often more robust than pure single shooting.

The next sections work through these formulations, starting with simultaneous methods, then sequential methods, and finally multiple shooting, before discussing how generic NLP solvers and specialized algorithms leverage the resulting structure in practice.

## Simultaneous Methods

What numerical structure appears when every state and action remains an
optimization variable and each transition becomes an equality constraint?

In the simultaneous (also called direct transcription or full discretization) approach, we keep the entire trajectory explicit and enforce the dynamics as equality constraints. Starting from the Bolza DOCP,

$$
\min_{\{\mathbf{x}_t,\mathbf{u}_t\}}\; c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t)
\quad\text{s.t.}\quad \mathbf{x}_{t+1} - \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t) = 0,\; t=1,\dots,T-1,
$$

collect all variables into a single vector

$$
\mathbf{z} := \begin{bmatrix}
\mathbf{x}_1^\top & \cdots & \mathbf{x}_T^\top & \mathbf{u}_1^\top & \cdots & \mathbf{u}_{T-1}^\top
\end{bmatrix}^\top \in \mathbb{R}^{n_z}.
$$

Path constraints typically apply only at selected times. Let $\mathscr{E}$ index additional equality constraints $g_i$ and $\mathscr{I}$ index inequality constraints $h_i$. For each constraint $i$, define the set of time indices $K_i \subseteq \{1,\dots,T\}$ where it is enforced (e.g., terminal constraints use $K_i = \{T\}$). The simultaneous transcription is the NLP

$$
\begin{aligned}
\min_{\mathbf{z}}\quad & F(\mathbf{z}) := c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t) \\
\text{s.t.}\quad & G(\mathbf{z}) = \begin{bmatrix}
\big[\, g_i(\mathbf{x}_k,\mathbf{u}_k) \big]_{i\in\mathscr{E},\, k\in K_i} \\
\big[\, \mathbf{x}_{t+1} - \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t) \big]_{t=1: T-1} \\
\mathbf{x}_1 - \mathbf{x}_\mathrm{init}
\end{bmatrix} = \mathbf{0}, \\
& H(\mathbf{z}) = \big[\, h_i(\mathbf{x}_k,\mathbf{u}_k) \big]_{i\in\mathscr{I},\, k\in K_i} \; \ge \; \mathbf{0},
\end{aligned}
$$

optionally with simple bounds $\mathbf{x}_{\mathrm{lb}} \le \mathbf{x}_t \le \mathbf{x}_{\mathrm{ub}}$ and $\mathbf{u}_{\mathrm{lb}} \le \mathbf{u}_t \le \mathbf{u}_{\mathrm{ub}}$ folded into $H$ or provided to the solver separately. For notational convenience, some constraints may not depend on $\mathbf{u}_k$ at times in $K_i$; the indexing still helps specify when each condition is active.

This direct transcription is attractive because it is faithful to the model and exposes sparsity. The Jacobian of $G$ has a block bi-diagonal structure induced by the dynamics, and the KKT matrix is sparse and structured. These properties are exploited by interior-point and SQP methods. The trade-off is size: with state dimension $n$ and control dimension $m$, the decision vector has $(T\!\cdot\!n) + ((T\!-
1)\cdot m)$ entries, and there are roughly $(T\!-
1)\cdot n$ dynamic equalities plus any path and boundary conditions. Techniques such as partial or full condensing eliminate state variables to reduce the equality set (at the cost of denser matrices), while keeping states explicit preserves sparsity and often improves robustness on long horizons and in the presence of state constraints.

Compared to alternatives, simultaneous methods avoid the long nonlinear dependency chains of single shooting and make it easier to impose state/path constraints. They can, however, demand more memory and per-iteration linear algebra, so practical performance hinges on exploiting sparsity and good initialization.

The same logic applies when selecting an optimizer. For small-scale problems, it is common to rely on general-purpose routines such as those in `scipy.optimize.minimize`. Derivative-free methods like Nelder–Mead require no gradients but scale poorly as dimensionality increases. Quasi-Newton schemes such as BFGS work well for moderate dimensions and can approximate gradients by finite differences, while large-scale trajectory optimization often calls for gradient-based constrained solvers such as interior-point or sequential quadratic programming methods that can exploit sparse Jacobians and benefit from automatic differentiation. Stochastic techniques, including genetic algorithms, simulated annealing, or particle swarm optimization, occasionally appear when gradients are unavailable, but their cost grows rapidly with dimension and they are rarely competitive for structured optimal control problems.

### Example: Nonlinear Cart-Pole Swing-Up

A cart carries a rigid pendulum whose angle is measured from the upright vertical. The cart can accelerate horizontally, but no actuator applies torque directly at the pendulum joint. Starting from the stable downward configuration, the task is to move the base so that the pendulum arrives upright while the cart returns near the center of a finite rail.

Let the state be $\mathbf{x}=(p,v,\theta,\omega)$, where $p$ and $v$ are the cart position and velocity, and $\theta$ and $\omega$ are the pendulum angle and angular velocity. A commanded horizontal acceleration $u$ produces the nonlinear dynamics

$$
\dot p = v, \qquad
\dot v = u, \qquad
\dot\theta = \omega, \qquad
\dot\omega = \frac{g}{\ell}\sin\theta
                 - \frac{u}{\ell}\cos\theta
                 - b\omega.
$$

The factor $-u\cos\theta/\ell$ identifies the action channel. Horizontal base motion couples into angular acceleration, and its sign and magnitude depend on the current configuration. A black-box optimizer could evaluate these equations without inspecting that term, but the term explains why the cart must first move away from its eventual resting position to build pendulum energy.

The numerical experiment uses a $4.5$ s horizon with $N=30$ zero-order-hold controls and a step size $h=0.15$ s. Fourth-order Runge--Kutta integration defines the discrete map $\mathbf{x}_{k+1}=F_h(\mathbf{x}_k,u_k)$. Both numerical formulations solve the same problem:

$$
\begin{aligned}
\min_{\mathbf{x}_{0:N},u_{0:N-1}}\quad
& h\sum_{k=0}^{N-1}\Bigl[
0.05p_k^2+0.01v_k^2+0.25(1-\cos\theta_k)
+0.01\omega_k^2+0.004u_k^2\Bigr] \\
& {}+20p_N^2+5v_N^2+120(1-\cos\theta_N)+12\omega_N^2 \\
\text{subject to}\quad
& \mathbf{x}_{k+1}=F_h(\mathbf{x}_k,u_k), \\
& \mathbf{x}_0=(0,0,\pi,0), \\
& |p_k|\leq 2.4,\quad |v_k|\leq 4,\quad
  |\omega_k|\leq 12,\quad |u_k|\leq 8.
\end{aligned}
$$

The periodic penalty $1-\cos\theta$ assigns the same terminal cost to angles that differ by a full revolution. Position and velocity penalties still require the cart to finish near rest, so rotating the pole through the top is not enough by itself.

Direct transcription retains all $31$ states and $30$ controls. It therefore optimizes over $154$ scalar variables and imposes $124$ scalar equalities, including the initial condition and one four-dimensional dynamics equation per step. The equality Jacobian is block banded because the residual at step $k$ depends only on $(\mathbf{x}_k,u_k,\mathbf{x}_{k+1})$.

The small demonstration below passes that Jacobian to SLSQP as a dense array. A large-scale direct solver would instead store and factor the same block-banded pattern sparsely. The formulation exposes sparsity, but exploiting it is a separate implementation choice.

```{admonition} Prediction before computation
:class: tip

Single shooting will reduce the decision vector from $154$ variables to $30$. Before examining the matched runs below, predict whether the smaller program must be easier to optimize. Identify which constraints become less explicit after the states are eliminated.
```


## Sequential Methods

Can eliminating the states reduce the nonlinear program without making the
resulting long simulation chain too sensitive to early actions?

The previous section showed how a discrete-time optimal control problem can be solved by treating all states and controls as decision variables and enforcing the dynamics as equality constraints. This produces a nonlinear program that can be passed to solvers such as `scipy.optimize.minimize` with the SLSQP method. For short horizons, this approach is straightforward and works well; the code stays close to the mathematical formulation.

It also has a real advantage: by keeping the states explicit and imposing the dynamics through constraints, we anchor the trajectory at multiple points. This extra structure helps stabilize the optimization, especially for long horizons where small deviations in early steps can otherwise propagate and cause the optimizer to drift or diverge. In that sense, this formulation is better conditioned and more robust than approaches that treat the dynamics implicitly.

The drawback is scale. As the horizon grows, the number of variables and constraints grows with it, and all are coupled by the dynamics. Each iteration of a sequential quadratic programming (SQP) or interior-point method requires building and factorizing large Jacobians and Hessians. These methods have been embedded in reinforcement learning and differentiable programming pipelines, through implicit layers or differentiable convex solvers, but the cost is significant. They remain serial, rely on repeated linear algebra factorizations, and are difficult to parallelize efficiently. When thousands of such problems must be solved inside a learning loop, the overhead becomes prohibitive.

This motivates an alternative that aligns with the computational model of machine learning. For deterministic dynamics, the equality constraints can be eliminated by making the states implicit. Instead of solving for both states and controls, we fix the initial state and roll the system forward under a candidate control sequence. State constraints can remain, but they become nonlinear functions of the entire preceding control sequence. This is the essence of **single shooting**.

The term "shooting" comes from the idea of *aiming and firing* a trajectory from the initial state: you pick a control sequence, integrate (or step) the system forward, and see where it lands. If the final state misses the target, you adjust the controls and try again: like adjusting the angle of a shot until it hits the mark. It is called **single** shooting because we compute the entire trajectory in one pass from the starting point, without breaking it into segments. Later, we will contrast this with **multiple shooting**, where the horizon is divided into smaller arcs that are optimized jointly to improve stability and conditioning.

The analogy with deep learning is also immediate: the control sequence plays the role of parameters, the rollout is a forward pass, and the cost is a scalar loss. Gradients can be obtained with reverse-mode automatic differentiation. In the single shooting formulation of the DOCP, the constrained program

$$
\min_{\mathbf{x}_{1:T},\,\mathbf{u}_{1:T-1}} J(\mathbf{x}_{1:T},\mathbf{u}_{1:T-1})
\quad\text{s.t.}\quad 
\mathbf{x}_{t+1}=\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t)
$$

collapses to

$$
\min_{\mathbf{u}_{1:T-1}}\;
c_T\!\bigl(\boldsymbol{\phi}_{T}(\mathbf{u}, \mathbf{x}_1)\bigr)
+\sum_{t=1}^{T-1} c_t\!\bigl(\boldsymbol{\phi}_{t}(\mathbf{u}, \mathbf{x}_1), \mathbf{u}_t\bigr),
\quad\text{s.t.}\quad
\mathbf{h}_t\!\bigl(\boldsymbol{\phi}_{t}(\mathbf{u},\mathbf{x}_1),\mathbf{u}_t\bigr)\geq 0,
\quad
\mathbf{u}_{\mathrm{lb}}\le\mathbf{u}_{t}\le\mathbf{u}_{\mathrm{ub}}.
$$

Here $\boldsymbol{\phi}_t$ denotes the state reached at time $t$ by recursively applying the dynamics to the previous state and current control. This recursion can be written as

$$
\boldsymbol{\phi}_{t+1}(\mathbf{u},\mathbf{x}_1)=
\mathbf{f}_{t}\!\bigl(\boldsymbol{\phi}_{t}(\mathbf{u},\mathbf{x}_1),\mathbf{u}_t\bigr),\qquad
\boldsymbol{\phi}_{1}=\mathbf{x}_1.
$$

Concretely, here is JAX-style pseudocode for defining `phi(u, x_0, t)` using `jax.lax.scan` with a zero-based time index:

```python
def phi(u_seq, x0, t):
    """Return \phi_t(u, x0) with 0-based t (\phi_0 = x0).

    u_seq: controls of length T (or T-1); only first t entries are used
    x0: initial state at time 0
    t: integer >= 0
    """
    if t <= 0:
        return x0

    def step(carry, u):
        x, t_idx = carry
        x_next = f(x, u, t_idx)
        return (x_next, t_idx + 1), None

    (x_t, _), _ = lax.scan(step, (x0, 0), u_seq[:t])
    return x_t
```


The pattern mirrors an RNN unroll: starting from an initial state ($\mathbf{x}^\star_1$) and a sequence of controls ($\mathbf{u}^*_{1:T-1}$), we propagate forward through the dynamics, updating the state at each step and accumulating cost along the way. This structural similarity is why single shooting often feels natural to practitioners with a deep learning background: the rollout is a forward pass, and gradients propagate backward through time exactly as in backpropagation through an RNN.

Algorithmically:

```{prf:algorithm} Single Shooting: Forward Unroll
:label: single-shooting-forward-unroll

**Inputs**: Initial state $\mathbf{x}_1$, horizon $T$, control bounds $\mathbf{u}_{\mathrm{lb}}, \mathbf{u}_{\mathrm{ub}}$, dynamics $\mathbf{f}_t$, costs $c_t$

**Output**: Optimal control sequence $\mathbf{u}^*_{1:T-1}$

1. Initialize $\mathbf{u}_{1:T-1}$ within bounds  
2. Define `ComputeTrajectoryAndCost`($\mathbf{u}, \mathbf{x}_1$):
    - $\mathbf{x} \leftarrow \mathbf{x}_1$, $J \leftarrow 0$
    - For $t = 1$ to $T-1$:
        - $J \leftarrow J + c_t(\mathbf{x}, \mathbf{u}_t)$
        - $\mathbf{x} \leftarrow \mathbf{f}_t(\mathbf{x}, \mathbf{u}_t)$
    - $J \leftarrow J + c_T(\mathbf{x})$
    - Return $J$
3. Solve $\min_{\mathbf{u}} J(\mathbf{u})$ subject to the control bounds and any state constraints evaluated along the rollout
4. Return $\mathbf{u}^*_{1:T-1}$
```

In JAX or PyTorch, this loop can be compiled and differentiated automatically. The control sequence plays the role of trainable parameters, while the simulated trajectory is the forward computation. Reverse-mode differentiation of that computation gives $\nabla J(\mathbf{u})$.

Single shooting is attractive for its simplicity and compatibility with differentiable programming, but it has limitations. Early controls influence every later state through a long product of dynamics Jacobians. This can make gradients poorly conditioned over long horizons. State constraints also lose their local sparse representation because each constrained state depends on all earlier controls. Formulations that keep selected states explicit, such as multiple shooting or collocation, shorten these dependency chains.

### Matched Swing-Up Comparison

The direct-transcription and single-shooting implementations below use the cart-pole problem stated above without changing the model, cost, limits, horizon, initial control guess, or nonlinear-programming solver. Only the decision variables and the representation of the dynamics differ.

```{code-cell} python
:tags: [remove-cell]

from pathlib import Path
import sys

code_directory = Path.cwd() / "code"
if str(code_directory) not in sys.path:
    sys.path.insert(0, str(code_directory))

from cartpole_control import (
    SwingUpScenario,
    format_swingup_metrics,
    make_open_loop_perturbation_figure,
    make_swingup_animation,
    make_swingup_figure,
    replay_open_loop_with_disturbance,
    solve_swingup_comparison,
)

swingup_scenario = SwingUpScenario()
swingup_results = solve_swingup_comparison(swingup_scenario)
```

```{code-cell} python
:label: fig-cartpole-formulations
:caption: Direct transcription and single shooting solve the same nonlinear cart-pole problem from the same initialization. Both reach the upright configuration and respect the matched limits, but they converge to different local solutions. Direct transcription retains 154 scalar variables and 124 local dynamics equalities; single shooting retains only the 30 controls and reconstructs every state by forward simulation.
:tags: [remove-input]

print(format_swingup_metrics(swingup_results))
make_swingup_figure(swingup_results, swingup_scenario)
```

Both solvers produce a successful open-loop swing-up. The direct formulation reaches a lower objective in this fixed run, while single shooting uses a much smaller decision vector. This numerical outcome does not establish that direct transcription always finds better solutions. It exposes a concrete trade-off: eliminating variables shortens the program but lengthens the dependency from an early control to the terminal cost and later constraints.

```{code-cell} python
:label: anim-cartpole-formulations
:caption: The two trajectories are generated by the same nonlinear RK4 plant. The pole starts downward and reaches the upright configuration while the cart remains inside the 2.4 m rail limits. Animation frames are computed from the optimized state trajectories; no browser-side simulator is used.
:tags: [remove-input]

from IPython.display import HTML, display
import matplotlib.pyplot as plt

swingup_animation = make_swingup_animation(swingup_results, swingup_scenario)
display(HTML(swingup_animation.to_jshtml()))
plt.close(swingup_animation._fig)
```

The comparison also separates optimization from feedback. Each optimizer returns one fixed control sequence for one assumed initial state. To test what that object can and cannot do, the next replay applies the direct-transcription controls twice. One realization follows the nominal model. The other receives an additional cart acceleration of $1\;\mathrm{m\,s^{-2}}$ for one $0.15$ s step at $t=2.1$ s, after which both realizations receive the same remaining commands.

```{code-cell} python
:label: fig-cartpole-open-loop-perturbation
:caption: A one-step unmodeled acceleration separates two realizations driven by the same open-loop controls. The nominal trajectory reaches normalized pole height $\cos\theta=1$; the disturbed trajectory finishes below the horizontal. The optimizer has produced a plan, not a rule that reacts to the observed state.
:tags: [remove-input]

open_loop_replay = replay_open_loop_with_disturbance(
    swingup_results["direct"],
    swingup_scenario,
)
make_open_loop_perturbation_figure(open_loop_replay)
```

Feedback changes the object being computed. A feedback controller maps the state observed after the disturbance to a new action. Model predictive control will obtain such a map by repeatedly solving trajectory problems, while dynamic programming will construct state-contingent decisions through the value function.

:::{dropdown} Inspect the shared nonlinear dynamics
```{literalinclude} code/cartpole_control.py
:language: python
:start-at: def cartpole_dynamics
:end-before: def rk4_step
:linenos:
```
:::

{download}`Download the complete cart-pole trajectory-optimization and control source <code/cartpole_control.py>`.



## In Between Sequential and Simultaneous

Can selected boundary states shorten those sensitivity paths while preserving
local simulation inside each segment?

```{admonition} Before reading on
:class: tip
Consider what happens to single shooting when the horizon $T$ is very large. What numerical difficulties might arise? Think about how small errors in early controls could propagate through the dynamics.
```

The two formulations we have seen so far lie at opposite ends. The **full discretization** approach keeps every state explicit and enforces the dynamics through equality constraints, which makes the structure clear but leads to a large optimization problem. At the other end, **single shooting** removes these constraints by simulating forward from the initial state, leaving only the controls as decision variables. That makes the problem smaller, but it also introduces a long and highly nonlinear dependency from the first control to the last state.

**Multiple shooting** sits in between. Instead of simulating the entire horizon in one shot, we divide it into smaller segments. For each segment, we keep its starting state as a decision variable and propagate forward using the dynamics for that segment. At the end, we enforce continuity by requiring that the simulated end state of one segment matches the decision variable for the next.

Formally, suppose the horizon of $T$ steps is divided into $K$ segments of length $L$ (with $T = K \cdot L$ for simplicity). We introduce:

* The controls for each step: $\mathbf{u}_{1:T-1}$.
* The state at the start of each segment: $\mathbf{x}_1,\dots,\mathbf{x}_K$.

Given $\mathbf{x}_k$ and the controls in its segment, we compute the predicted terminal state by simulating forward:

$$
\hat{\mathbf{x}}_{k+1} = \Phi(\mathbf{x}_k,\mathbf{u}_{\text{segment }k}),
$$

where $\Phi$ represents $L$ applications of the dynamics. Continuity constraints enforce:

$$
\mathbf{x}_{k+1} - \hat{\mathbf{x}}_{k+1} = 0, \qquad k=1,\dots,K-1.
$$

The resulting nonlinear program looks like this:

$$
\begin{aligned}
\min_{\{\mathbf{x}_k,\mathbf{u}_t\}} \quad &
c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t) \\
\text{subject to} \quad &
\mathbf{x}_{k+1} - \Phi(\mathbf{x}_k,\mathbf{u}_{\text{segment }k}) = 0,\quad k = 1,\dots,K-1, \\
& \mathbf{u}_{\mathrm{lb}} \le \mathbf{u}_t \le \mathbf{u}_{\mathrm{ub}}, \\
& \text{boundary conditions on } \mathbf{x}_1 \text{ and } \mathbf{x}_K.
\end{aligned}
$$

Compared to the full NLP, we no longer introduce every intermediate state as a variable, only the anchors at segment boundaries. Inside each segment, states are reconstructed by simulation. Compared to single shooting, these anchors break the long dependency chain that makes optimization unstable: gradients only have to travel across $L$ steps before they hit a decision variable, rather than the entire horizon. This is the same reason why exploding or vanishing gradients appear in deep recurrent networks: when the chain is too long, information either dies out or blows up. Multiple shooting shortens the chain and improves conditioning.

By adjusting the number of segments $K$, we can interpolate between the two extremes: $K = 1$ gives single shooting, while $K = T$ recovers the full direct NLP. In practice, a moderate number of segments often strikes a good balance between robustness and complexity.


```{code-cell} python
:tags: [hide-input]

#  label: fig-ocp-multiple-shooting
#  caption: Multiple shooting ballistic BVP: the code produces an animation (and optional static plot) that shows how segment defects shrink while steering the projectile to the target.

%config InlineBackend.figure_format = 'retina'
"""
Multiple Shooting as a Boundary-Value Problem (BVP) for a Ballistic Trajectory
-----------------------------------------------------------------------------
We solve for the initial velocities (and total flight time) so that the terminal
position hits a target, enforcing continuity between shooting segments.
"""

import numpy as np
import matplotlib.pyplot as plt

# Apply book style
try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
except (ImportError, OSError):
    pass  # Use matplotlib defaults
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from IPython.display import HTML, display

# -----------------------------
# Physical parameters
# -----------------------------
g = 9.81          # gravity (m/s^2)
m = 1.0           # mass (kg)
drag_coeff = 0.1  # quadratic drag coefficient


def dynamics(t, state):
    """Ballistic dynamics with quadratic drag. state = [x, y, vx, vy]."""
    x, y, vx, vy = state
    v = np.hypot(vx, vy)
    drag_x = -drag_coeff * v * vx / m if v > 0 else 0.0
    drag_y = -drag_coeff * v * vy / m if v > 0 else 0.0
    dx  = vx
    dy  = vy
    dvx = drag_x
    dvy = drag_y - g
    return np.array([dx, dy, dvx, dvy])


def flow(y0, h):
    """One-segment flow map Φ(y0; h): integrate dynamics over duration h."""
    sol = solve_ivp(dynamics, (0.0, h), y0, method="RK45", rtol=1e-7, atol=1e-9)
    return sol.y[:, -1], sol

# -----------------------------
# Multiple-shooting BVP residuals
# -----------------------------

def residuals(z, K, x_init, x_target):
    """
    Unknowns z = [vx0, vy0, H, y1(4), y2(4), ..., y_{K-1}(4)]  (total len = 3 + 4*(K-1))
    We define y0 from x_init and (vx0, vy0). Each segment has duration h = H/K.
    Residual vector stacks:
      - initial position constraints: y0[:2] - x_init[:2]
      - continuity: y_{k+1} - Φ(y_k; h) for k=0..K-2
      - terminal position constraint at end of last segment: Φ(y_{K-1}; h)[:2] - x_target[:2]
    """
    n = 4
    vx0, vy0, H = z[0], z[1], z[2]
    if H <= 0:
        # Strongly penalize nonpositive durations to keep solver away
        return 1e6 * np.ones(2 + 4*(K-1) + 2)

    h = H / K

    # Build list of segment initial states y_0..y_{K-1}
    ys = []
    y0 = np.array([x_init[0], x_init[1], vx0, vy0], dtype=float)
    ys.append(y0)
    if K > 1:
        rest = z[3:]
        y_internals = rest.reshape(K-1, n)
        ys.extend(list(y_internals))  # y1..y_{K-1}

    res = []

    # Initial position must match exactly
    res.extend(ys[0][:2] - x_init[:2])

    # Continuity across segments
    for k in range(K-1):
        yk = ys[k]
        yk1_pred, _ = flow(yk, h)
        res.extend(ys[k+1] - yk1_pred)

    # Terminal position at the end of last segment equals target
    y_last_end, _ = flow(ys[-1], h)
    res.extend(y_last_end[:2] - x_target[:2])

    # Optional soft "stay above ground" at knots (kept gentle)
    # res.extend(np.minimum(0.0, np.array([y[1] for y in ys])).ravel())

    return np.asarray(res)

# -----------------------------
# Solve BVP via optimization on 0.5*||residuals||^2
# -----------------------------

def solve_bvp_multiple_shooting(K=5, x_init=np.array([0., 0.]), x_target=np.array([10., 0.])):
    """
    K: number of shooting segments.
    x_init: initial position (x0, y0). Initial velocities are unknown.
    x_target: desired terminal position (xT, yT) at time H (unknown).
    """
    # Heuristic initial guesses:
    dx = x_target[0] - x_init[0]
    dy = x_target[1] - x_init[1]
    H0 = max(0.5, dx / 5.0)  # guess ~ 5 m/s horizontal
    vx0_0 = dx / H0
    vy0_0 = (dy + 0.5 * g * H0**2) / H0  # vacuum guess

    # Intentionally disconnected internal knots to visualize defect shrinkage
    internals = []
    for k in range(1, K):  # y1..y_{K-1}
        xk = x_init[0] + (dx * k) / K
        yk = x_init[1] + (dy * k) / K + 2.0  # offset to create mismatch
        internals.append(np.array([xk, yk, 0.0, 0.0]))
    internals = np.array(internals) if K > 1 else np.array([])

    z0 = np.concatenate(([vx0_0, vy0_0, H0], internals.ravel()))

    # Variable bounds: H > 0, keep velocities within a reasonable range
    # Use wide bounds to let the solver work; tune if needed.
    lb = np.full_like(z0, -np.inf, dtype=float)
    ub = np.full_like(z0,  np.inf, dtype=float)
    lb[2] = 1e-2  # H lower bound
    # Optional velocity bounds
    lb[0], ub[0] = -50.0, 50.0
    lb[1], ub[1] = -50.0, 50.0

    # Objective and callback for L-BFGS-B
    def objective(z):
        r = residuals(z, K,
                      np.array([x_init[0], x_init[1], 0., 0.]),
                      np.array([x_target[0], x_target[1], 0., 0.]))
        return 0.5 * np.dot(r, r)

    iterate_history = []
    def cb(z):
        iterate_history.append(z.copy())

    bounds = list(zip(lb.tolist(), ub.tolist()))
    sol = minimize(objective, z0, method='L-BFGS-B', bounds=bounds,
                   callback=cb, options={'maxiter': 300, 'ftol': 1e-12})

    return sol, iterate_history

# -----------------------------
# Reconstruct and plot (optional static figure)
# -----------------------------

def reconstruct_and_plot(sol, K, x_init, x_target):
    n = 4
    vx0, vy0, H = sol.x[0], sol.x[1], sol.x[2]
    h = H / K

    ys = []
    y0 = np.array([x_init[0], x_init[1], vx0, vy0])
    ys.append(y0)
    if K > 1:
        internals = sol.x[3:].reshape(K-1, n)
        ys.extend(list(internals))

    # Integrate each segment and stitch
    traj_x, traj_y = [], []
    for k in range(K):
        yk = ys[k]
        yend, seg = flow(yk, h)
        traj_x.extend(seg.y[0, :].tolist() if k == 0 else seg.y[0, 1:].tolist())
        traj_y.extend(seg.y[1, :].tolist() if k == 0 else seg.y[1, 1:].tolist())

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(traj_x, traj_y, '-', label='Multiple-shooting solution')
    ax.plot([x_init[0]], [x_init[1]], 'go', label='Start')
    ax.plot([x_target[0]], [x_target[1]], 'r*', ms=12, label='Target')
    total_pts = len(traj_x)
    for k in range(1, K):
        idx = int(k * total_pts / K)
        ax.axvline(traj_x[idx], color='k', ls='--', alpha=0.3, lw=1)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Multiple Shooting BVP (K={K})   H={H:.3f}s   v0=({vx0:.2f},{vy0:.2f}) m/s')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()

    # Report residual norms
    res = residuals(sol.x, K, np.array([x_init[0], x_init[1], 0., 0.]), np.array([x_target[0], x_target[1], 0., 0.]))
    print(f"\nFinal residual norm: {np.linalg.norm(res):.3e}")
    print(f"vx0={vx0:.4f} m/s, vy0={vy0:.4f} m/s, H={H:.4f} s")

# -----------------------------
# Create JS animation for notebooks
# -----------------------------

def create_animation_progress(iter_history, K, x_init, x_target):
    """Return a JS animation (to_jshtml) showing defect shrinkage across segments."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Apply book style
    try:
        import scienceplots
        plt.style.use(['science', 'notebook'])
    except (ImportError, OSError):
        pass  # Use matplotlib defaults

    n = 4

    def unpack(z):
        vx0, vy0, H = z[0], z[1], z[2]
        ys = [np.array([x_init[0], x_init[1], vx0, vy0])]
        if K > 1 and len(z) > 3:
            internals = z[3:].reshape(K-1, n)
            ys.extend(list(internals))
        return H, ys

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.set_xlabel('Segment index (normalized time)')
    ax.set_ylabel('y (m)')
    ax.set_title('Multiple Shooting: Defect Shrinkage (Fixed Boundaries)')
    ax.grid(True, alpha=0.3)

    # Start/target markers at fixed indices
    ax.plot([0], [x_init[1]], 'go', label='Start')
    ax.plot([K], [x_target[1]], 'r*', ms=12, label='Target')
    # Vertical dashed lines at boundaries
    for k in range(1, K):
        ax.axvline(k, color='k', ls='--', alpha=0.35, lw=1)
    ax.legend(loc='best')

    # Pre-create line artists
    colors = plt.cm.plasma(np.linspace(0, 1, K))
    segment_lines = [ax.plot([], [], '-', color=colors[k], lw=2, alpha=0.9)[0] for k in range(K)]
    connector_lines = [ax.plot([], [], 'r-', lw=1.4, alpha=0.75)[0] for _ in range(K-1)]

    text_iter = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    def animate(i):
        idx = min(i, len(iter_history)-1)
        z = iter_history[idx]
        H, ys = unpack(z)
        h = H / K

        all_y = [x_init[1], x_target[1]]
        total_defect = 0.0
        for k in range(K):
            yk = ys[k]
            yend, seg = flow(yk, h)
            # Map local time to [k, k+1]
            t_local = seg.t
            x_vals = k + (t_local / t_local[-1])
            y_vals = seg.y[1, :]
            segment_lines[k].set_data(x_vals, y_vals)
            all_y.extend(y_vals.tolist())
            if k < K-1:
                y_next = ys[k+1]
                # Vertical connector at boundary x=k+1
                connector_lines[k].set_data([k+1, k+1], [yend[1], y_next[1]])
                total_defect += abs(y_next[1] - yend[1])

        # Fixed x-limits in index space
        ax.set_xlim(-0.1, K + 0.1)
        ymin, ymax = min(all_y), max(all_y)
        margin_y = 0.10 * max(1.0, ymax - ymin)
        ax.set_ylim(ymin - margin_y, ymax + margin_y)

        text_iter.set_text(f'Iterate {idx+1}/{len(iter_history)}  |  Sum vertical defect: {total_defect:.3e}')
        return segment_lines + connector_lines + [text_iter]

    anim = FuncAnimation(fig, animate, frames=len(iter_history), interval=600, blit=False, repeat=True)
    plt.tight_layout()
    js_anim = anim.to_jshtml()
    plt.close(fig)
    return js_anim


def main():
    # Problem definition
    x_init = np.array([0.0, 0.0])      # start at origin
    x_target = np.array([10.0, 0.0])   # hit ground at x=10 m
    K = 6                               # number of shooting segments

    sol, iter_hist = solve_bvp_multiple_shooting(K=K, x_init=x_init, x_target=x_target)
    # Optionally show static reconstruction (commented for docs cleanliness)
    # reconstruct_and_plot(sol, K, x_init, x_target)

    # Animate progression (defect shrinkage across segments) and display as JS
    js_anim = create_animation_progress(iter_hist, K, x_init, x_target)
    display(HTML(js_anim))


if __name__ == "__main__":
    main()
```





### Example: Hydro Cascade Scheduling with Physical Routing

The ballistic boundary-value problem couples consecutive segments of one trajectory. A hydroelectric cascade adds a second form of coupling: actions taken upstream alter the inflows seen downstream after a travel delay. Multiple shooting exposes both forms through local ODE integrations, temporal continuity defects, and inter-reach routing constraints.

The hydro-reservoir model in [](finite-horizon-dp.md) uses a discrete-time abstraction in which precipitation enters as a noisy inflow. That abstraction is useful for learning and control design, but it omits much of the physical behavior of rivers and dams. Here we use a more detailed setup inspired by {cite:p}`Savorgnan2011`. We consider a series of dams arranged in a cascade, where the actions taken upstream influence downstream levels with a delay. The amount of power produced depends on the water flow through the turbines and the head (the vertical distance between the reservoir surface and the turbine outlet). The larger the head, the more potential energy is available for conversion into electricity, and the higher the power output.

To capture these effects, we follow a modeling approach inspired by the Saint-Venant equations, which describe how water levels and flows evolve in open channels. Instead of solving the full PDEs, we use a reduced model that approximates each dammed section of river (called a reach) as a lumped system governed by an ordinary differential equation. The main variable of interest is the water level $h_r(t)$, which changes over time depending on how much water enters, how much is discharged through the turbines $q_r(t)$, and how much is spilled $s_r(t)$. The mass balance for reach $r$ is written as:

$$
\frac{d h_r(t)}{dt} = \frac{1}{A_r} \left( z_r(t) - q_r(t) - s_r(t) \right),
$$

where $A_r$ is the surface area of the reservoir, assumed constant. The inflow $z_r(t)$ to a reach either comes from nature (for the first dam), or from the upstream turbine and spill discharge, delayed by a travel time $\tau_{r-1}$:

$$
z_1(t) = \text{inflow}(t), \qquad
z_r(t) = q_{r-1}(t - \tau_{r-1}) + s_{r-1}(t - \tau_{r-1}), \quad \text{for } r > 1.
$$

Power generation at each reach depends on how much water is discharged and the available head:

$$
P_r(t) = \rho g \eta \, q_r(t) \, H_r(h_r(t)),
$$

where $\rho$ is water density, $g$ is gravitational acceleration, $\eta$ is turbine efficiency, and $H_r(h_r(t))$ denotes the head as a function of the water level. In some models, the head is approximated as the difference between the current level and a fixed tailwater height (the water level downstream of the dam, after it has passed through the turbine).

The operator's goal is to meet a target generation profile $P^\text{ref}(t)$, such as one dictated by a market dispatch or load-following constraint. This leads to an objective that minimizes the deviation from the target over the full horizon:

$$
\min_{\{q_r(t), s_r(t)\}} \int_0^T \left( \sum_{r=1}^R P_r(t) - P^\text{ref}(t) \right)^2 dt.
$$

In practice, this is combined with operational constraints: turbine capacity $0 \le q_r(t) \le \bar{q}_r$, spillway limits $0 \le s_r(t) \le \bar{s}_r$, and safe level bounds $h_r^{\min} \le h_r(t) \le h_r^{\max}$. Depending on the use case, one may also penalize spill to encourage water conservation, or penalize fast changes in levels for ecological reasons.

The reaches are coupled across space and time. An upstream reach cannot simply act in isolation: if the operator wants reach $r$ to produce power at a specific time, the water must be released by reach $r-1$ sufficiently in advance. This coordination is further complicated by delays, nonlinearities in head-dependent power, and limited storage capacity.

We solve the problem using **multiple shooting**. Each reach is divided into local simulation segments over short time windows. Within each segment, the dynamics are integrated forward using the ODEs, and continuity constraints are added to ensure that the water levels match across segment boundaries. At the same time, the inflows passed from upstream reaches must arrive at the right time and be consistent with previous decisions. In discrete time, this gives rise to a set of state-update equations:

$$
h_r^{k+1} = h_r^k + \Delta t \cdot \frac{1}{A_r}(z_r^k - q_r^k - s_r^k),
$$

with delays handled by shifting $z_r^k$ according to the appropriate travel time. These constraints are enforced as part of a nonlinear program, alongside the power tracking objective and control bounds.

Compared with a single-reservoir inflow-outflow model, the cascade adds delayed coupling constraints. Upstream reservoirs can store water in anticipation of future needs, while downstream dams adjust their output to match arrivals and avoid overflows. The resulting schedule coordinates the entire system against the demand profile.

```{code-cell} python
:tags: [hide-input]
:label: fig-trajectories-hydro-multiple-shooting
:caption: Multiple shooting coordinates reservoir levels, turbine discharges, routed inflows, and total generation across a three-reach hydroelectric cascade.

%config InlineBackend.figure_format = 'retina'
# Instrumented MSD hydro demo with heterogeneity + diagnostics
# - Breaks symmetry to avoid trivial identical plots
# - Adds rich diagnostics to explain flat levels and equalities
#
# This cell runs end-to-end and shows plots + tables.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Apply book style
try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
except (ImportError, OSError):
    pass  # Use matplotlib defaults
from dataclasses import dataclass
from typing import Tuple
from scipy.optimize import minimize
from math import sqrt
import warnings

# ---------- Model ----------

g = 9.81  # m/s^2

@dataclass
class ReachParams:
    L: float
    W: float
    k_b: float
    S_b: float
    k_t: float
    @property
    def A_surf(self) -> float:
        return self.L * self.W

def smooth_relu(x, eps=1e-9):
    return 0.5*(x + np.sqrt(x*x + eps))

def q_bypass(H, rp: ReachParams):
    H_eff = smooth_relu(H)
    return rp.k_b * rp.S_b * np.sqrt(2*g*H_eff)

def muskingum_coeffs(K: float, X: float, dt: float) -> Tuple[float, float, float]:
    D  = 2.0*K*(1.0 - X) + dt
    C0 = (dt - 2.0*K*X) / D
    C1 = (dt + 2.0*K*X) / D
    C2 = (2.0*K*(1.0 - X) - dt) / D
    return C0, C1, C2

def integrate_interval(H0, u, z, dt, nsub, rp: ReachParams):
    """Forward Euler. Returns Hend, avg_qout."""
    h = dt/nsub
    H = H0
    qsum = 0.0
    for _ in range(nsub):
        qb = q_bypass(H, rp)
        qout = u + qb
        dHdt = (z - qout) / rp.A_surf
        H += h*dHdt
        qsum += qout
    return H, qsum/nsub

def shapes(M,N): return (M*(N+1), M*N, M*N)

def unpack(x, M, N):
    nH, nu, nz = shapes(M,N)
    H = x[:nH].reshape(M,N+1)
    u = x[nH:nH+nu].reshape(M,N)
    z = x[nH+nu:nH+nu+nz].reshape(M,N)
    return H,u,z

def pack(H,u,z): return np.concatenate([H.ravel(), u.ravel(), z.ravel()])

# ---------- Problem builder ----------

def make_params_hetero(M):
    """Heterogeneous reaches to break symmetry."""
    # Widths, spillway areas, and power coeffs vary by reach
    W_list = np.linspace(80, 140, M)         # m
    L_list = np.full(M, 4000.0)              # m
    S_b_list = np.linspace(14.0, 20.0, M)    # m^2
    k_t_list = np.linspace(7.5, 8.5, M)      # power coeff
    k_b_list = np.linspace(0.55, 0.65, M)    # spill coeff
    return [ReachParams(L=float(L_list[i]), W=float(W_list[i]),
                        k_b=float(k_b_list[i]), S_b=float(S_b_list[i]),
                        k_t=float(k_t_list[i])) for i in range(M)]

def build_demo(M=3, N=12, dt=900.0, seed=0, hetero=True):
    rng = np.random.default_rng(seed)
    params = make_params_hetero(M) if hetero else [ReachParams(4000.0, 100.0, 0.6, 18.26, 8.0) for _ in range(M)]

    # initial levels (heterogeneous)
    H0 = np.array([17.0, 16.7, 17.3][:M])

    H_ref = np.array([17.0, 16.9, 17.1][:M]) if hetero else np.full(M, 17.0)
    H_bounds = (16.0, 18.5)
    u_bounds = (40.0, 160.0)

    Qin_base = 300.0
    Qin_ext = Qin_base + 30.0*np.sin(2*np.pi*np.arange(N)/N)  # stronger swing

    Pref_raw = 60.0 + 15.0*np.sin(2*np.pi*(np.arange(N)-2)/N)

    # default Muskingum parameters per link (M-1 links)
    if M > 1:
        K_list = list(np.linspace(1800.0, 2700.0, M-1))
        X_list = [0.2]*(M-1)
    else:
        K_list = []
        X_list = []

    return dict(params=params, H0=H0, H_ref=H_ref, H_bounds=H_bounds,
                u_bounds=u_bounds, Qin_ext=Qin_ext, Pref_raw=Pref_raw,
                dt=dt, N=N, M=M, nsub=10,
                muskingum=dict(K=K_list, X=X_list))

# ---------- Objective / constraints / helpers ----------

def compute_total_power(H,u,params):
    M,N = u.shape
    Pn = np.zeros(N)
    for n in range(N):
        for i in range(M):
            Pn[n] += params[i].k_t * u[i,n] * H[i,n]
    return Pn

def decompose_objective(x, data, Pref, wP, wH, wDu):
    H,u,z = unpack(x, data["M"], data["N"])
    params, H_ref = data["params"], data["H_ref"]
    track = np.sum((compute_total_power(H,u,params)-Pref)**2)
    lvl   = np.sum((H[:,:-1]-H_ref[:,None])**2)
    du    = np.sum((u[:,1:]-u[:,:-1])**2)
    return dict(track=wP*track, lvl=wH*lvl, du=wDu*du, raw=dict(track=track,lvl=lvl,du=du))

def make_objective(data, Pref, wP=8.0, wH=0.02, wDu=1e-4):
    params, H_ref, N, M = data["params"], data["H_ref"], data["N"], data["M"]
    def obj(x):
        H,u,z = unpack(x,M,N)
        return (
            wP*np.sum((compute_total_power(H,u,params)-Pref)**2)
            + wH*np.sum((H[:,:-1]-H_ref[:,None])**2)
            + wDu*np.sum((u[:,1:]-u[:,:-1])**2)
        )
    return obj, dict(wP=wP,wH=wH,wDu=wDu)

def make_constraints(data):
    params, H0, Qin_ext, dt, N, M, nsub = (
        data["params"], data["H0"], data["Qin_ext"], data["dt"], data["N"], data["M"], data["nsub"]
    )
    cons = []
    def init_fun(x):
        H,u,z = unpack(x,M,N); return H[:,0]-H0
    cons.append({'type':'eq','fun':init_fun})
    def dyn_fun(x):
        H,u,z = unpack(x,M,N)
        res=[]
        for i in range(M):
            for n in range(N):
                Hend, _ = integrate_interval(H[i,n], u[i,n], z[i,n], dt, nsub, params[i])
                res.append(H[i,n+1]-Hend)
        return np.array(res)
    cons.append({'type':'eq','fun':dyn_fun})
    def coup_fun(x):
        H,u,z = unpack(x,M,N)
        res=[]
        # First reach is exogenous inflow per interval
        for n in range(N):
            res.append(z[0,n]-Qin_ext[n])
        # Downstream links: Muskingum routing
        K_list = data.get("muskingum", {}).get("K", [])
        X_list = data.get("muskingum", {}).get("X", [])
        for i in range(1,M):
            # Seed condition for z[i,0]
            _, I0 = integrate_interval(H[i-1,0], u[i-1,0], z[i-1,0], dt, nsub, params[i-1])
            res.append(z[i,0] - I0)
            # Coefficients
            Ki = K_list[i-1] if i-1 < len(K_list) else 1800.0
            Xi = X_list[i-1] if i-1 < len(X_list) else 0.2
            C0, C1, C2 = muskingum_coeffs(Ki, Xi, dt)
            # Recursion over intervals
            for n in range(N-1):
                # upstream interval-average outflows for n and n+1
                _, I_n   = integrate_interval(H[i-1,n],   u[i-1,n],   z[i-1,n],   dt, nsub, params[i-1])
                _, I_np1 = integrate_interval(H[i-1,n+1], u[i-1,n+1], z[i-1,n+1], dt, nsub, params[i-1])
                res.append(z[i,n+1] - (C0*I_np1 + C1*I_n + C2*z[i,n]))
        return np.array(res)
    cons.append({'type':'eq','fun':coup_fun})
    return cons

def make_bounds(data):
    Hmin,Hmax = data["H_bounds"]
    umin,umax = data["u_bounds"]
    M,N = data["M"], data["N"]
    nH,nu,nz = shapes(M,N)
    lb = np.empty(nH+nu+nz); ub = np.empty_like(lb)
    lb[:nH]=Hmin; ub[:nH]=Hmax
    lb[nH:nH+nu]=umin; ub[nH:nH+nu]=umax
    lb[nH+nu:]=0.0; ub[nH+nu:]=2000.0
    return list(zip(lb,ub))

def residuals(x, data):
    params, H0, Qin_ext, dt, N, M, nsub = (
        data["params"], data["H0"], data["Qin_ext"], data["dt"], data["N"], data["M"], data["nsub"]
    )
    H,u,z = unpack(x, M, N)
    dyn = np.zeros((M,N)); coup = np.zeros((M,N))
    for i in range(M):
        for n in range(N):
            Hend, qavg = integrate_interval(H[i,n], u[i,n], z[i,n], dt, nsub, params[i])
            dyn[i,n] = H[i,n+1] - Hend
            if i == 0:
                coup[i,n] = z[i,n] - Qin_ext[n]
            else:
                # Muskingum residual, align on current index using n and n-1
                Ki = data.get("muskingum", {}).get("K", [1800.0]*(M-1))[i-1]
                Xi = data.get("muskingum", {}).get("X", [0.2]*(M-1))[i-1]
                C0, C1, C2 = muskingum_coeffs(Ki, Xi, dt)
                if n == 0:
                    coup[i,n] = 0.0
                else:
                    _, I_nm1 = integrate_interval(H[i-1,n-1], u[i-1,n-1], z[i-1,n-1], dt, nsub, params[i-1])
                    _, I_n   = integrate_interval(H[i-1,n],   u[i-1,n],   z[i-1,n],   dt, nsub, params[i-1])
                    coup[i,n] = z[i,n] - (C0*I_n + C1*I_nm1 + C2*z[i,n-1])
    return dyn, coup

# ---------- Feasible initial guess with hetero controls ----------

def feasible_initial_guess(data):
    """Feasible x0 with nontrivial u by setting u at mid + per-reach pattern, then integrating to define H,z."""
    M,N,dt,nsub = data["M"], data["N"], data["dt"], data["nsub"]
    params = data["params"]
    umin,umax = data["u_bounds"]
    Qin_ext = data["Qin_ext"]

    # pattern to break symmetry
    base = 0.5*(umin+umax)
    phase = np.linspace(0, np.pi/2, M)
    tgrid = np.arange(N)
    u_pattern = np.array([base + 25*np.sin(2*np.pi*(tgrid/N) + ph) for ph in phase])
    u_pattern = np.clip(u_pattern, umin, umax)

    H = np.zeros((M, N+1)); u = np.zeros((M, N)); z = np.zeros((M, N))
    H[:,0] = data["H0"]
    # Set controls from pattern first
    for i in range(M):
        u[i,:] = u_pattern[i,:]

    # First reach: exogenous inflow, integrate forward and record outflow averages
    qavg_up = np.zeros((M, N))
    for n in range(N):
        z[0,n] = Qin_ext[n]
        Hend, qavg = integrate_interval(H[0,n], u[0,n], z[0,n], dt, nsub, params[0])
        H[0,n+1] = Hend
        qavg_up[0,n] = qavg

    # Downstream reaches with Muskingum routing
    K_list = data.get("muskingum", {}).get("K", [1800.0]*(M-1))
    X_list = data.get("muskingum", {}).get("X", [0.2]*(M-1))
    for i in range(1,M):
        Ki = K_list[i-1] if i-1 < len(K_list) else 1800.0
        Xi = X_list[i-1] if i-1 < len(X_list) else 0.2
        C0, C1, C2 = muskingum_coeffs(Ki, Xi, dt)
        I = qavg_up[i-1,:]
        # seed
        z[i,0] = I[0]
        # propagate recursively over time
        for n in range(N-1):
            z[i,n+1] = C0*I[n+1] + C1*I[n] + C2*z[i,n]
        # integrate levels for reach i using routed inflow
        for n in range(N):
            Hend, qavg = integrate_interval(H[i,n], u[i,n], z[i,n], dt, nsub, params[i])
            H[i,n+1] = Hend
            qavg_up[i,n] = qavg
    return pack(H,u,z)

def scale_pref(Pref_raw, x0, data):
    H,u,z = unpack(x0, data["M"], data["N"])
    P0 = compute_total_power(H,u,data["params"])
    s = max(np.mean(P0),1e-6)/max(np.mean(Pref_raw),1e-6)
    return Pref_raw*s, P0

def run_demo(show: bool = True, save_path: str | None = 'hydro.png', verbose: bool = False):
    """Build, solve, and render the hydro demo.

    Parameters
    ----------
    show : bool
        If True, displays the matplotlib figure via plt.show().
    save_path : str | None
        If provided, saves the figure to this path.
    verbose : bool
        If True, prints diagnostic information.

    Returns
    -------
    matplotlib.figure.Figure | None
        Returns the Figure when show is False; otherwise returns None.
    """
    # ---------- Solve ----------
    data = build_demo(M=3, N=16, dt=900.0, hetero=True)
    x0 = feasible_initial_guess(data)
    Pref, P0 = scale_pref(data["Pref_raw"], x0, data)

    objective, weights = make_objective(data, Pref, wP=8.0, wH=0.02, wDu=5e-4)
    # Suppress noisy SciPy warning about delta_grad during quasi-Newton updates
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"delta_grad == 0.0",
            category=UserWarning,
            module=r"scipy\.optimize\.\_differentiable_functions",
        )
        res = minimize(
            fun=objective,
            x0=x0,
            method='trust-constr',
            bounds=make_bounds(data),
            constraints=make_constraints(data),
            options=dict(maxiter=1000, disp=verbose),
        )

    H,u,z = unpack(res.x, data["M"], data["N"])
    P = compute_total_power(H,u,data["params"])
    dyn_res, coup_res = residuals(res.x, data)

    # ---------- Diagnostics ----------
    if verbose:
        terms = decompose_objective(res.x, data, Pref, **weights)
        print("\n=== Objective decomposition ===")
        print({k: float(v) if not isinstance(v, dict) else {kk: float(vv) for kk,vv in v.items()} for k,v in terms.items()})

        print("\n=== Constraint residuals (max |.|) ===")
        print("dyn:", float(np.max(np.abs(dyn_res)))), print("coup:", float(np.max(np.abs(coup_res))))

        # Muskingum coefficient sanity and residuals
        if data.get("M", 1) > 1:
            K_list = data.get("muskingum", {}).get("K", [])
            X_list = data.get("muskingum", {}).get("X", [])
            coef_checks = []
            mean_abs_res = []
            for i in range(1, data["M"]):
                Ki = K_list[i-1] if i-1 < len(K_list) else 1800.0
                Xi = X_list[i-1] if i-1 < len(X_list) else 0.2
                C0, C1, C2 = muskingum_coeffs(Ki, Xi, data["dt"])
                coef_checks.append(dict(link=i, sum=float(C0+C1+C2), min_coef=float(min(C0,C1,C2))))
                # compute mean abs residual for this link
                res_vals = []
                for n in range(data["N"]-1):
                    _, I_n   = integrate_interval(H[i-1,n],   u[i-1,n],   z[i-1,n],   data["dt"], data["nsub"], data["params"][i-1])
                    _, I_np1 = integrate_interval(H[i-1,n+1], u[i-1,n+1], z[i-1,n+1], data["dt"], data["nsub"], data["params"][i-1])
                    res_vals.append(float(abs(z[i,n+1] - (C0*I_np1 + C1*I_n + C2*z[i,n]))))
                mean_abs_res.append(dict(link=i, mean_abs=float(np.mean(res_vals))))
            print("\n=== Muskingum coeff checks (sum, min_coef) ===")
            print(coef_checks)
            print("=== Muskingum mean |residual| per link ===")
            print(mean_abs_res)

    # Per-interval diagnostic table for each reach (kept for debugging but unused here)
    def interval_table(i):
        rp = data["params"][i]
        rows = []
        for n in range(data["N"]):
            qb = q_bypass(H[i,n], rp)
            net = z[i,n] - (u[i,n] + qb)
            dH = data["dt"]*net/rp.A_surf
            rows.append(dict(interval=n, Hn=H[i,n], Hn1=H[i,n+1], u=u[i,n], z=z[i,n], qb=qb, net_flow=net, dH_pred=dH))
        return pd.DataFrame(rows)

    # summary and tables available to callers if needed
    tables = [interval_table(i) for i in range(data["M"])]
    summary = pd.DataFrame([
        dict(reach=i+1,
             H_mean=float(np.mean(H[i])), H_std=float(np.std(H[i])),
             u_mean=float(np.mean(u[i])), u_std=float(np.std(u[i])),
             z_mean=float(np.mean(z[i])), z_std=float(np.std(z[i])))
        for i in range(data["M"])
    ])

    # ---------- Plots ----------
    M,N = data["M"], data["N"]
    t_nodes = np.arange(N+1)
    t = np.arange(N)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Hydroelectric System Optimization Results', fontsize=16)

    ax1 = axes[0, 0]
    for i in range(M):
        ax1.plot(t_nodes, H[i], marker='o', label=f'Reach {i+1}')
    ax1.set_xlabel("Node n"); ax1.set_ylabel("H [m]"); ax1.set_title("Water Levels")
    ax1.grid(True); ax1.legend()

    ax2 = axes[0, 1]
    for i in range(M):
        ax2.step(t, u[i], where='post', label=f'Reach {i+1}')
    ax2.set_xlabel("Interval n"); ax2.set_ylabel("u [m³/s]"); ax2.set_title("Turbine Discharge")
    ax2.grid(True); ax2.legend()

    ax3 = axes[1, 0]
    for i in range(M):
        ax3.step(t, z[i], where='post', label=f'Reach {i+1}')
    ax3.set_xlabel("Interval n"); ax3.set_ylabel("z [m³/s]"); ax3.set_title("Inflow (Coupling)")
    ax3.grid(True); ax3.legend()

    ax4 = axes[1, 1]
    ax4.plot(t, P0, marker='s', label="Power @ x0")
    ax4.plot(t, P, marker='o', label="Power @ optimum")
    ax4.plot(t, Pref, marker='x', label="Scaled Pref")
    ax4.set_xlabel("Interval n"); ax4.set_ylabel("Power units"); ax4.set_title("Power Tracking")
    ax4.legend(); ax4.grid(True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        return None
    return fig


# Run the demo directly when loaded in a notebook cell
run_demo(show=True, save_path=None, verbose=False)

```


The figure shows the result of a multiple-shooting optimization applied to a three-reach hydroelectric cascade. The time horizon is discretized into 16 intervals, and SciPy's `trust-constr` solver is used to find a feasible control sequence that satisfies mass balance, turbine and spillway limits, and Muskingum-style routing dynamics. Each reach integrates its own local ODE. Shooting defects link reservoir levels across time, while separate Muskingum constraints link routed flows between reaches.

The top-left panel shows the water levels in each reservoir. We observe that upstream reservoirs tend to increase their levels ahead of discharge events, building potential energy before releasing water downstream. The top-right panel shows turbine discharges for each reach. These vary smoothly and are temporally coordinated across the system. The bottom-right panel compares the total generation to a synthetic demand profile, which is generated by a sum of time-shifted sigmoids and normalized to be feasible given turbine capacities. The optimized schedule (orange) tracks this demand closely, while the initial guess (blue) lags behind. The bottom-left panel plots the routed inflows between reaches, which display the expected lag and smoothing effects from Muskingum routing. The interplay between these plots shows how the system anticipates, stores, and routes water to meet time-varying generation targets within physical and operational limits.

The ballistic and hydro examples use the same numerical structure at different scales: integrate locally, expose states at segment boundaries, and drive every continuity defect to zero. We now return to the first-order optimality conditions of the underlying discrete-time program.

## Summary and Outlook

Direct transcription keeps states and controls explicit and exposes sparse
dynamics constraints. Single shooting eliminates the states but couples early
actions to every later quantity through one rollout. Multiple shooting keeps
selected boundary states, trading additional variables for shorter sensitivity
paths and sparse continuity defects.

All three formulations assume a discrete transition map. How can a
continuous-time trajectory and its differential equations be represented by a
finite set of decision variables and algebraic constraints? [Continuous-time
transcription and collocation](continuous-time-collocation.md) answer that
question with nodal polynomial representations.

## Exercises

:::{exercise} Single shooting implementation
:label: ex-trajectories-single-shooting

Consider the scalar system $x_{t+1} = x_t + u_t$ with $x_1 = 0$ and objective:

$$
J = x_T^2 + \sum_{t=1}^{T-1} u_t^2.
$$

**(a)** Express $x_T$ as a function of the controls $u_1, \ldots, u_{T-1}$.

**(b)** Substitute into $J$ to obtain an unconstrained objective in the controls only.

**(c)** Implement single shooting in Python/JAX to minimize $J$ for $T = 10$. Use gradient descent with a learning rate of 0.1 for 100 iterations. Report the optimal controls and final cost.

:::

:::{solution} ex-trajectories-single-shooting
:class: dropdown

**(a)** $x_T = \sum_{t=1}^{T-1} u_t$.

**(b)** $J(u) = \left(\sum_{t=1}^{T-1} u_t\right)^2 + \sum_{t=1}^{T-1} u_t^2$.

**(c)** Sample code:
```python
import jax.numpy as jnp
from jax import grad

def objective(u):
    x_T = jnp.sum(u)
    return x_T**2 + jnp.sum(u**2)

T = 10
u = jnp.zeros(T - 1)
for _ in range(100):
    u = u - 0.1 * grad(objective)(u)

print(f"Optimal u: {u}, Cost: {objective(u):.4f}")
```
The optimal controls should be approximately equal and negative, with total cost near $0$.
:::

---

:::{exercise} Multiple shooting segments
:label: ex-trajectories-multiple-shooting

Using the same problem as Exercise 5, implement multiple shooting with $K = 3$ segments.

**(a)** Define the segment boundaries and the continuity defects.

**(b)** Set up the NLP with decision variables $[x_1, x_4, x_7, u_1, \ldots, u_9]$ (for $T=10$, with segments of length 3).

**(c)** Compare the convergence behavior to single shooting. Does multiple shooting require fewer iterations to reach the same tolerance?

:::

:::{solution} ex-trajectories-multiple-shooting
:class: dropdown

The defects are $d_k = x_{k+1}^{\text{simulated}} - x_{k+1}^{\text{variable}}$ at segment boundaries. You can minimize $J + \rho \sum_k \|d_k\|^2$ for large $\rho$ (penalty method) or use a constrained solver. Multiple shooting typically converges faster for longer horizons because the optimization landscape is better conditioned.
:::

---

## Self-checks

:::{exercise} Count the decisions
:label: ex-trajectories-check-1

A single-shooting problem has horizon $T$ and scalar controls. How many optimization variables remain after eliminating the states, and what computation couples an early control to the terminal cost?
:::

:::{solution} ex-trajectories-check-1
:class: dropdown

There are $T$ control variables. Forward simulation couples every early control to all later states and therefore to the terminal cost.
:::

:::{exercise} Shooting trade-off
:label: ex-trajectories-check-2

Why can multiple shooting be easier to optimize than single shooting even though it introduces more decision variables?
:::

:::{solution} ex-trajectories-check-2
:class: dropdown

Intermediate states break a long sensitive rollout into shorter segments. The resulting continuity constraints are sparse, and derivatives need not propagate through the full horizon in one chain.
:::
