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

# Discrete-Time Trajectory Optimization

In the previous chapter, we examined different ways to represent dynamical systems: continuous versus discrete time, deterministic versus stochastic, fully versus partially observable, and even simulation-based views such as agent-based or programmatic models. Our focus was on the **structure of models**: how they capture evolution, uncertainty, and information.

In this chapter, we turn to what makes these models useful for **decision-making**. The goal is no longer just to describe how a system behaves, but to leverage that description to **compute actions over time**. This doesn’t mean the model prescribes actions on its own. Rather, it provides the scaffolding for optimization: given a model and an objective, we can derive the control inputs that make the modeled system behave well according to a chosen criterion. This is the essence of an **optimal control problem**.

## Discrete-Time Optimal Control Problems (DOCPs)

Consider a system described by a **state** $\mathbf{x}_t \in \mathbb{R}^n$, summarizing everything needed to predict its evolution. At each stage $t$, we can influence the system through a **control input** $\mathbf{u}_t \in \mathbb{R}^m$. The dynamics specify how the state evolves:

$$
\mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t, \mathbf{u}_t),
$$

where $\mathbf{f}_t$ may be nonlinear or time-varying. We assume the initial state $\mathbf{x}_1$ is known.

The goal is to pick a sequence of controls $\mathbf{u}_1,\dots,\mathbf{u}_{T-1}$ that makes the trajectory desirable. But desirable in what sense? That depends on an **objective function**, which often includes two components:

$$
\text{(i) stage cost: } c_t(\mathbf{x}_t,\mathbf{u}_t), \qquad \text{(ii) terminal cost: } c_T(\mathbf{x}_T).
$$

The stage cost reflects ongoing penalties—energy, delay, risk. The terminal cost measures the value (or cost) of ending in a particular state. **Together, these give the canonical Bolza form**:

$$
\begin{aligned}
\text{minimize} \quad & c_T(\mathbf{x}_T) + \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t) \\
\text{subject to} \quad & \mathbf{x}_{t+1} = \mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t), \ t=1,\dots,T-1.
\end{aligned}
$$

Written this way, it may seem obvious that the decision variables are the controls $\mathbf{u}_t$. After all, in most intuitive descriptions of control, we think of choosing inputs to influence the system. But notice that in the program above, the entire state trajectory also appears as a set of variables, linked to the controls by the dynamics constraints. This is intentional: it reflects one way of writing the problem that makes the constraints explicit.

Why introduce $\mathbf{x}_t$ as decision variables if they can be simulated forward from the controls? Many readers hesitate here, and the question is natural: *If the model is deterministic and $\mathbf{x}_1$ is known, why not pick $\mathbf{u}_{1:T-1}$ and compute $\mathbf{x}_{2:T}$ on the fly?* That instinct leads to **single shooting**, a method we will return to shortly.

Already in this formulation, though, we see an important theme: **the structure of the problem matters**. Ignoring it can make our life much harder. The reason is twofold:

* **Dimensionality grows with the horizon.** For a horizon of length $T$, the program has roughly $(T-1)(m+n)$ decision variables.
* **Temporal coupling.** Each control affects all future states and costs. The feasible set is not a simple box but a narrow manifold defined by the dynamics.

Together, these features explain why specialized methods exist and why the way we write the problem influences the algorithms we can use. Whether we keep states explicit or eliminate them through forward simulation determines not just the problem size, but also its conditioning and the trade-offs between robustness and computational effort.

### Variants: Lagrange and Mayer Problems

The Bolza form is general enough to cover most situations, but two common special cases are worth noting:

* **Lagrange problem (no terminal cost)**
  If the objective only accumulates stage costs:

$$
\min_{\mathbf{u}_{1:T-1}} \sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t).
$$

Example: *Energy minimization for a delivery drone*. The concern is total battery use, regardless of the final position.

* **Mayer problem (terminal cost only)**
  If the objective depends only on the final state:

$$
\min_{\mathbf{u}_{1:T-1}} c_T(\mathbf{x}_T).
$$

Example: *Satellite orbital transfer*. The only goal is to reach a specified orbit, no matter the fuel spent along the way.

These distinctions matter when deriving optimality conditions, but conceptually they fit in the same framework: the system evolves over time, and we choose controls to shape the trajectory.

### Reducing to Mayer Form by State Augmentation

Although Bolza, Lagrange, and Mayer problems look different, they are equivalent in expressive power. Any problem with running costs can be rewritten as a Mayer problem (one whose objective depends only on the final state) through a simple trick: **augment the state with a running sum of costs**.

The idea is straightforward. Introduce a new variable, $y_t$, that keeps track of the cumulative cost so far. At each step, we update this running sum along with the system state:

$$
\tilde{\mathbf{x}}_{t+1} =
\begin{pmatrix}
\mathbf{f}_t(\mathbf{x}_t,\mathbf{u}_t) \\
y_t + c_t(\mathbf{x}_t,\mathbf{u}_t)
\end{pmatrix},
$$

where $\tilde{\mathbf{x}}_t = (\mathbf{x}_t, y_t)$. The terminal cost then becomes:

$$
\tilde{c}_T(\tilde{\mathbf{x}}_T) = c_T(\mathbf{x}_T) + y_T.
$$

The overall effect is that the explicit sum $\sum_{t=1}^{T-1} c_t(\mathbf{x}_t,\mathbf{u}_t)$ disappears from the objective and is captured implicitly by the augmented state. This lets us write every optimal control problem in Mayer form.

Why do this? Two reasons. First, it often simplifies **mathematical derivations**, as we will see later when deriving necessary conditions. Second, it can **streamline algorithmic implementation**: instead of writing separate code paths for Mayer, Lagrange, and Bolza problems, we can reduce everything to one canonical form. That said, this “one size fits all” approach isn’t always best in practice—specialized formulations can sometimes be more efficient computationally, especially when the running cost has simple structure.


The unifying theme is that a DOCP may look like a generic NLP on paper, but its structure matters. Ignoring that structure often leads to impractical solutions, whereas formulations that expose sparsity and respect temporal coupling allow modern solvers to scale effectively. In the following sections, we will examine how these choices play out in practice through single shooting, multiple shooting, and collocation methods, and why different formulations strike different trade-offs between robustness and computational effort.

# Numerical Methods for Solving DOCPs

## Simultaneous Methods

Once a discrete-time optimal control problem (DOCP) has been formulated, the question becomes how to compute a solution. 
If we collect all states and controls into a single vector $\mathbf{z}$, the problem can be written in the canonical form of a nonlinear program (NLP):

$$
\begin{aligned}
\text{minimize} \quad & f(\mathbf{z}) \\
\text{subject to} \quad & \mathbf{g}(\mathbf{z}) \le 0, \\
& \mathbf{h}(\mathbf{z}) = 0,
\end{aligned}
$$

where $\mathbf{h}$ enforces the dynamics and $\mathbf{g}$ encodes path or box constraints. Standard solvers such as IPOPT or SNOPT accept this formulation directly, which might suggest that trajectory optimization is simply a matter of handing the problem to a black box.

In practice, however, this simplicity is misleading. Optimal control problems have structural characteristics that make them considerably harder than generic nonlinear programs of comparable size. The number of variables grows with the planning horizon, since both states and controls appear explicitly. A horizon of length $T$ with state dimension $n$ and control dimension $m$ already implies on the order of $(T-1)(n+m)$ decision variables. More importantly, these variables are tightly linked by the dynamics: each decision affects all subsequent states and costs. This temporal coupling means the feasible set is highly structured, and naive formulations that ignore this often lead to poor numerical conditioning and slow convergence.

These properties influence not only the difficulty of the problem but also the choice of algorithm. For instance, one approach is to collapse the problem to control inputs only and compute states by simulation; this single-shooting formulation eliminates equality constraints but introduces strong nonlinear dependencies on the controls, making the optimization sensitive to initial guesses. At the other extreme, methods such as multiple shooting or direct collocation retain states as explicit variables and enforce dynamics through constraints, leading to larger problems but with a sparsity structure that specialized solvers can exploit efficiently. The decision between these strategies is therefore not cosmetic: it determines whether the solver can take advantage of the problem’s structure.

The same logic applies when selecting an optimizer. For small-scale problems, it is common to rely on general-purpose routines such as those in `scipy.optimize.minimize`. Derivative-free methods like Nelder–Mead require no gradients but scale poorly as dimensionality increases. Quasi-Newton schemes such as BFGS work well for moderate dimensions and can approximate gradients by finite differences, while large-scale trajectory optimization often calls for gradient-based constrained solvers such as interior-point or sequential quadratic programming methods that can exploit sparse Jacobians and benefit from automatic differentiation. Stochastic techniques, including genetic algorithms, simulated annealing, or particle swarm optimization, occasionally appear when gradients are unavailable, but their cost grows rapidly with dimension and they are rarely competitive for structured optimal control problems.

<!-- ### On the Choice of Optimizer

Although the code example uses SLSQP, many alternatives exist. `scipy.optimize.minimize` provides a menu of options, and each has implications for speed, robustness, and scalability:

* **Derivative-free methods** such as Nelder–Mead avoid gradients altogether. They are attractive when gradients are unavailable or noisy, but they scale poorly with dimension.
* **Quasi-Newton methods** like BFGS approximate gradients by finite differences. They work well for moderate-scale problems and often outperform derivative-free schemes when the objective is smooth.
* **Gradient-based constrained solvers** such as interior-point or SQP methods exploit derivatives—exact or automatic—and are typically the most efficient for large structured problems like trajectory optimization.

Beyond these, **stochastic optimizers** occasionally appear in practice, especially when gradients are unreliable or the loss landscape is rugged. Random search is the simplest example, while genetic algorithms, simulated annealing, and particle swarm optimization introduce mechanisms for global exploration at the cost of significant computational effort.

Which method to choose depends on the context: problem size, availability of derivatives, and computational resources. When automatic differentiation is accessible, first-order methods like L-BFGS or Adam often dominate, particularly for single-shooting formulations where the objective is smooth and unconstrained except for simple bounds. This is why researchers with a machine learning background tend to gravitate toward these techniques: they integrate seamlessly with existing frameworks and run efficiently on GPUs. -->

### Example: Direct Solution to the Eco-cruise Problem

Many modern vehicles include features that aim to improve energy efficiency without requiring extra effort from the driver. One such feature is Eco-Cruise. Unlike traditional cruise control, which keeps the car at a fixed speed regardless of conditions, Eco-Cruise adjusts speed within small margins to reduce energy consumption. The reasoning is straightforward: holding speed up a hill by applying full throttle uses more energy than allowing the car to slow slightly and regain speed later. Some systems go further by using map data, anticipating slopes and curves to plan ahead. These ideas are no longer experimental; several manufacturers already deploy predictive cruise systems based on navigation input.

The setup we will use is slightly idealized, but not unrealistic. It assumes that the driver provides a destination and an acceptable time target, something that most navigation systems already require. With that information, the controller can decide how fast to go and when to accelerate while ensuring the trip remains on schedule. Framing the problem in this way allows us to cast Eco-Cruise as a trajectory optimization exercise and to explore the structure of a discrete-time optimal control problem.

Consider a 1 km segment of road that must be completed in exactly 60 seconds. We divide this horizon into 60 steps of one second each. At step $t$, the state consists of the cumulative distance $s_t$ and the speed $v_t$. The control input is the longitudinal acceleration $u_t$. With a time step of one second, the dynamics are written as

$$
s_{t+1} = s_t + v_t, \qquad
v_{t+1} = v_t + u_t.
$$

The trip starts from rest, so $s_1 = 0$ and $v_1 = 0$, and it must end at $s_{T+1} = 1000$ m with $v_{T+1} = 0$.

Energy consumption depends on both acceleration and speed. Rather than model the details of rolling resistance, drivetrain losses, and aerodynamics, we adopt a simple quadratic approximation. Each stage incurs a cost

$$
c_t(v_t, u_t) = \tfrac{1}{2}\beta u_t^2 + \tfrac{1}{2}\gamma v_t^2,
$$

where the first term penalizes strong accelerations and the second discourages high cruising speed. Reasonable values are $\beta = 1.0$ and $\gamma = 0.1$. The objective is to minimize the sum of these stage costs across the horizon:

$$
\min \sum_{t=1}^{T} \bigl( \tfrac{\beta}{2}u_t^2 + \tfrac{\gamma}{2}v_t^2 \bigr).
$$

The optimization must also respect physical limits. Speeds must remain between zero and $20\ \text{m/s}$ (about 72 km/h), and accelerations are bounded by $|u_t| \le 3\ \text{m/s}^2$ for comfort and safety.


The complete formulation is

$$
\begin{aligned}
\min_{\{s_t,v_t,u_t\}} \ & \sum_{t=1}^{T} \bigl( \tfrac{\beta}{2}u_t^2 + \tfrac{\gamma}{2}v_t^2 \bigr) \\
\text{subject to}\ & s_{t+1}-s_t-v_t = 0,\ \ v_{t+1}-v_t-u_t = 0,\ t=1,\dots,T, \\
& s_1 = 0,\ v_1 = 0,\ s_{T+1} = 1000,\ v_{T+1} = 0, \\
& 0 \le v_t \le 20,\ \ |u_t|\le 3.
\end{aligned}
$$

#### Solution

Once the objective and constraints are expressed as Python functions, the problem can be passed to a generic optimizer with very little extra work. Here is a direct implementation using `scipy.optimize.minimize` with the SLSQP method:

```{code-cell} ipython3
:load: code/eco-cruise.py
:tags: [remove-input, remove-output]
```

```{glue:figure} eco_cruise_figure
:figwidth: 100%
:name: "fig-eco-cruise"

Eco-Cruise optimization results showing the comparison between energy-efficient and naive trajectory approaches.
```

``````{tab-set}
:tags: [full-width]

`````{tab-item} Visualization
```{raw} html
<script src="_static/iframe-modal.js"></script>
<div id="eco-cruise-container"></div>
<script>
createIframeModal({
  containerId: 'eco-cruise-container',
  iframeSrc: '_static/eco-cruise-demo.html',
  title: 'Eco-Cruise Optimization Visualization',
  aspectRatio: '200%',
  maxWidth: '1400px',
  maxHeight: '900px'
});
</script>
`````

`````{tab-item} Code
```{literalinclude} code/eco-cruise.py
:language: python
```
`````
``````

The function `scipy.optimize.minimize` expect three things: an objective function that returns a scalar cost, a set of constraints grouped as equality or inequality functions, and bounds on individual variables. Everything else is about bookkeeping.

The first step is to gather all decision variables—positions, speeds, and accelerations—into a single vector $\mathbf{z}$. Helper routines like `unpack` then slice this vector back into its components so that the rest of the code reads naturally. The objective function mirrors the analytical form of the cost: it sums quadratic penalties on speeds and accelerations across the horizon.

Dynamics and boundary conditions appear as equality constraints. Each entry in `dynamics` enforces one of the discrete-time equations

$$
s_{t+1} - s_t - v_t = 0,\qquad
v_{t+1} - v_t - u_t = 0,
$$

while `boundary` pins down the start and end conditions. Together, these ensure that any candidate solution corresponds to a physically consistent trajectory.

Bounds serve two purposes: they impose physical limits on speed and acceleration and keep the otherwise unbounded position variables within a large but finite range. This prevents the optimizer from exploring meaningless regions of the search space during intermediate iterations.

Finally, an initial guess is constructed by interpolating a straight line for the position, assigning a constant speed, and setting accelerations to zero. This is not intended to be optimal; it simply gives the solver a feasible starting point close enough to the constraint manifold to converge quickly.

Once these components are in place, the call to `minimize` does the rest. Internally, SLSQP linearizes the constraints, builds a quadratic subproblem, and iterates until both the Karush–Kuhn–Tucker conditions and the stopping tolerances are met. From the user’s perspective, the heavy lifting reduces to providing functions that compute costs and residuals—everything else is handled by the solver.



## Sequential Methods

The previous section showed how a discrete-time optimal control problem can be solved by treating all states and controls as decision variables and enforcing the dynamics as equality constraints. This produces a nonlinear program that can be passed to solvers such as `scipy.optimize.minimize` with the SLSQP method. For short horizons, this approach is straightforward and works well; the code stays close to the mathematical formulation.

It also has a real advantage: by keeping the states explicit and imposing the dynamics through constraints, we anchor the trajectory at multiple points. This extra structure helps stabilize the optimization, especially for long horizons where small deviations in early steps can otherwise propagate and cause the optimizer to drift or diverge. In that sense, this formulation is better conditioned and more robust than approaches that treat the dynamics implicitly.

The drawback is scale. As the horizon grows, the number of variables and constraints grows with it, and all are coupled by the dynamics. Each iteration of a sequential quadratic programming (SQP) or interior-point method requires building and factorizing large Jacobians and Hessians. These methods have been embedded in reinforcement learning and differentiable programming pipelines—through implicit layers or differentiable convex solvers—but the cost is significant. They remain serial, rely on repeated linear algebra factorizations, and are difficult to parallelize efficiently. When thousands of such problems must be solved inside a learning loop, the overhead becomes prohibitive.

This motivates an alternative that aligns better with the computational model of machine learning. If the dynamics are deterministic and state constraints are absent (or reducible to simple bounds on controls), we can eliminate the equality constraints altogether by making the states implicit. Instead of solving for both states and controls, we fix the initial state and roll the system forward under a candidate control sequence. This is the essence of **single shooting**.

The term “shooting” comes from the idea of *aiming and firing* a trajectory from the initial state: you pick a control sequence, integrate (or step) the system forward, and see where it lands. If the final state misses the target, you adjust the controls and try again—like adjusting the angle of a shot until it hits the mark. It is called **single** shooting because we compute the entire trajectory in one pass from the starting point, without breaking it into segments. Later, we will contrast this with **multiple shooting**, where the horizon is divided into smaller arcs that are optimized jointly to improve stability and conditioning.

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
\qquad
\mathbf{u}_{\mathrm{lb}}\le\mathbf{u}_{t}\le\mathbf{u}_{\mathrm{ub}}.
$$

Here $\boldsymbol{\phi}_t$ denotes the state reached at time $t$ by recursively applying the dynamics to the previous state and current control. This recursion can be written as

$$
\boldsymbol{\phi}_{t+1}(\mathbf{u},\mathbf{x}_1)=
\mathbf{f}_{t}\!\bigl(\boldsymbol{\phi}_{t}(\mathbf{u},\mathbf{x}_1),\mathbf{u}_t\bigr),\qquad
\boldsymbol{\phi}_{1}=\mathbf{x}_1.
$$


Concretely, implementing this amounts to running a **for loop**—or, in frameworks like JAX or TensorFlow, using a `scan` operator. The pattern mirrors an RNN unroll: starting from an initial state (\$\mathbf{x}*1\$) and a sequence of controls (\$\mathbf{u}*{1\:T-1}\$), we propagate forward through the dynamics, updating the state at each step and accumulating cost along the way. This structural similarity is why single shooting often feels natural to practitioners with a deep learning background: the rollout is a forward pass, and gradients propagate backward through time exactly as in backpropagation through an RNN.

Algorithmically:

```{prf:algorithm} Single Shooting: Forward Unroll
:label: single-shooting-forward-unroll

**Inputs**: Initial state $\mathbf{x}_1$, horizon $T$, control bounds $\mathbf{u}_{\mathrm{lb}}, \mathbf{u}_{\mathrm{ub}}$, dynamics $\mathbf{f}_t$, costs $c_t$

**Output**: Optimal control sequence $\mathbf{u}^*_{1:T-1}$

1. Initialize $\mathbf{u}_{1:T-1}$ within bounds  
2. Define `ComputeTrajectoryAndCost($\mathbf{u}, \mathbf{x}_1$)`:
    - $\mathbf{x} \leftarrow \mathbf{x}_1$, $J \leftarrow 0$
    - For $t = 1$ to $T-1$:
        - $J \leftarrow J + c_t(\mathbf{x}, \mathbf{u}_t)$
        - $\mathbf{x} \leftarrow \mathbf{f}_t(\mathbf{x}, \mathbf{u}_t)$
    - $J \leftarrow J + c_T(\mathbf{x})$
    - Return $J$
3. Solve $\min_{\mathbf{u}} J(\mathbf{u})$ subject to $\mathbf{u}_{\mathrm{lb}} \le \mathbf{u}_t \le \mathbf{u}_{\mathrm{ub}}$
4. Return $\mathbf{u}^*_{1:T-1}$
```

In JAX or PyTorch, this loop can be JIT-compiled and differentiated automatically. Any gradient-based optimizer—L-BFGS, Adam, even SGD—can be applied, making the pipeline look very much like training a neural network. In effect, we are “backpropagating through the world model” when computing \$\nabla J(\mathbf{u})\$.

Single shooting is attractive for its simplicity and compatibility with differentiable programming, but it has limitations. The absence of intermediate constraints makes it sensitive to initialization and prone to numerical instability over long horizons. When state constraints or robustness matter, formulations that keep states explicit—such as multiple shooting or collocation—become preferable. These trade-offs are the focus of the next section.

<!-- 
```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show code demonstration"
:  code_prompt_hide: "Hide code demonstration"
:load: code/single_shooting_unrolled.py
``` -->

## In Between Sequential and Simultaneous

The two formulations we have seen so far lie at opposite ends. The **direct NLP approach** keeps every state explicit and enforces the dynamics through equality constraints, which makes the structure clear but leads to a large optimization problem. At the other end, **single shooting** removes these constraints by simulating forward from the initial state, leaving only the controls as decision variables. That makes the problem smaller, but it also introduces a long and highly nonlinear dependency from the first control to the last state.

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

Compared to the full NLP, we no longer introduce every intermediate state as a variable—only the anchors at segment boundaries. Inside each segment, states are reconstructed by simulation. Compared to single shooting, these anchors break the long dependency chain that makes optimization unstable: gradients only have to travel across $L$ steps before they hit a decision variable, rather than the entire horizon. This is the same reason why exploding or vanishing gradients appear in deep recurrent networks: when the chain is too long, information either dies out or blows up. Multiple shooting shortens the chain and improves conditioning.

By adjusting the number of segments $K$, we can interpolate between the two extremes: $K = 1$ gives single shooting, while $K = T$ recovers the full direct NLP. In practice, a moderate number of segments often strikes a good balance between robustness and complexity.

```{code-cell} ipython3
:load: code/generate_multiple_shooting_trajectory.py
:tags: [remove-input, remove-output]
```

```{glue:} multiple_shooting_output
```

``````{tab-set}
:tags: [full-width]

`````{tab-item} Visualization
```{raw} html
<script src="_static/iframe-modal.js"></script>
<div id="multiple-shooting-container"></div>
<script>
createIframeModal({
  containerId: 'multiple-shooting-container',
  iframeSrc: '_static/multiple-shooting-demo.html',
  title: 'Multiple Shooting Visualization',
  aspectRatio: '150%',
  maxWidth: '1400px',
  maxHeight: '900px'
});
</script>
```
`````

`````{tab-item} Code
```{literalinclude} code/generate_multiple_shooting_trajectory.py
:language: python
```
`````
``````

