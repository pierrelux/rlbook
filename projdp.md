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


# Projection Methods for Functional Equations

The Bellman optimality equation $\mathrm{L}v = v$ is a functional equation—an equation where the unknown is an entire function rather than a finite-dimensional vector. When the state space is continuous or very large, we cannot represent the value function exactly on a computer. We must instead work with finite-dimensional approximations. This motivates projection methods, a general framework for transforming infinite-dimensional problems into tractable finite-dimensional ones.

**The central challenge**: even if we restrict our search to a finite-dimensional subspace of functions, we still need to verify that our candidate solution $\hat{v}$ satisfies the equation. For a true solution, the **residual function** $R(s) = \mathrm{L}\hat{v}(s) - \hat{v}(s)$ should equal zero at every state $s$ in our (potentially infinite) state space. But we cannot check infinitely many conditions.

**The projection method solution**: replace "the residual is zero everywhere" with a feasible requirement that we can verify computationally. We restrict our search to functions $\hat{v}(s) = \sum_{i=1}^n a_i \varphi_i(s)$ for some basis $\{\varphi_1, \ldots, \varphi_n\}$, and find coefficients $a$ that make the residual "small" according to a chosen criterion. The criterion determines the projection method. 

## The General Framework

Consider an operator equation of the form

$$
\mathscr{N}(f) = 0,
$$

where $\mathscr{N}: B_1 \to B_2$ is a continuous operator between complete normed vector spaces $B_1$ and $B_2$. For the Bellman equation, we have $\mathscr{N}(v) = \mathrm{L}v - v$, so that solving $\mathscr{N}(v) = 0$ is equivalent to finding the fixed point $v = \mathrm{L}v$.

The projection method approach consists of several conceptual steps that transform this infinite-dimensional problem into a finite-dimensional one.

### Step 1: Choose a Finite-Dimensional Approximation Space

We begin by selecting a basis $\Phi = \{\varphi_1, \varphi_2, \ldots, \varphi_n\}$ and approximating the unknown function as a linear combination:

$$
\hat{f}(x) = \sum_{i=1}^n a_i \varphi_i(x).
$$

The choice of basis functions $\varphi_i$ is crucial and problem-dependent. Common choices include:
- **Polynomials**: For smooth problems, we might use Chebyshev polynomials or other orthogonal polynomial families
- **Splines**: For problems where we expect the solution to have regions of different smoothness
- **Radial basis functions**: For high-dimensional problems where tensor product methods become intractable

The number of basis functions $n$ determines the flexibility of our approximation. In practice, we start with small $n$ and increase it until the approximation quality is satisfactory. The only unknowns now are the coefficients $a = (a_1, \ldots, a_n)$.

### Step 2: Define the Residual Function

Since we are approximating $f$ with $\hat{f}$, the operator $\mathscr{N}$ will generally not vanish exactly. Instead, we obtain a **residual function**:

$$
R(x; a) = \mathscr{N}(\hat{f}(\cdot; a))(x).
$$

For a true solution, this residual would be identically zero everywhere: $R(x; a) = 0$ for all $x$ in the domain. For our approximation, we aim to make it as close to the zero function as possible. **The residual measures how far our candidate solution is from satisfying the equation at each point $x$.** 

Think of $R(\cdot; a)$ as living in an infinite-dimensional function space—you can imagine it as a "vector" with one component $R(x; a)$ for each point $x$ in the domain. In finite dimensions, we would check if a vector is zero by verifying that each component equals zero. Here, we face an impossible task: we cannot verify that $R(x; a) = 0$ at every single point. The projection conditions we introduce next provide different feasible ways to measure "how zero" this function is.

### Step 3: Choose Projection Conditions

The heart of the projection method is choosing how to verify that $R(\cdot; a)$ is "close to zero." Since we cannot check $R(x; a) = 0$ at every point, we must select a feasible criterion. 

**The unifying framework**: Nearly all projection methods work by choosing $n$ **test functions** $\{p_1, \ldots, p_n\}$ and requiring:

$$
\langle R(\cdot; a), p_i \rangle = \int_{\mathcal{S}} R(x; a) p_i(x) w(x) dx = 0, \quad i = 1, \ldots, n,
$$

for some weight function $w(x)$. This yields $n$ equations to determine the $n$ coefficients in $a$. **The different projection methods are distinguished entirely by their choice of test functions** $p_i$. If the residual has zero projection against all our test functions, we declare it "close enough" to the zero function.

**Intuition from finite dimensions**: In $\mathbb{R}^n$, to verify a vector $\mathbf{r}$ is zero, we could check if $\langle \mathbf{r}, \mathbf{e}_i \rangle = 0$ for each standard basis vector $\mathbf{e}_i$. If $\mathbf{r}$ is orthogonal to all coordinate directions, it must be the zero vector. For functions in infinite dimensions, we cannot test against all directions, so we choose $n$ representative test functions and verify orthogonality against them.

Let us examine the standard choices of test functions and what they reveal about the residual:

#### Galerkin Method: Test Against the Basis

The Galerkin method chooses test functions $p_i = \varphi_i$, the same basis functions used to approximate $\hat{f}$:

$$
\langle R(\cdot; a), \varphi_i \rangle = 0, \quad i = 1, \ldots, n.
$$

To understand what this means, recall that in finite dimensions, two vectors are orthogonal when their inner product is zero. For functions, $\langle R, \varphi_i \rangle = \int R(x) \varphi_i(x) w(x) dx = 0$ expresses the same concept: $R$ and $\varphi_i$ are orthogonal as functions. But there's more to this than just testing against individual basis functions.

Consider our approximation space $\text{span}\{\varphi_1, \ldots, \varphi_n\}$ as an $n$-dimensional subspace within the infinite-dimensional space of all functions. Any function $g$ in this space can be written as $g = \sum_{i=1}^n c_i \varphi_i$ for some coefficients $c_i$. If the residual $R$ is orthogonal to all basis functions $\varphi_i$, then by linearity of the inner product, for any such function $g$:

$$
\langle R, g \rangle = \left\langle R, \sum_{i=1}^n c_i \varphi_i \right\rangle = \sum_{i=1}^n c_i \langle R, \varphi_i \rangle = 0.
$$

This shows that $R$ is orthogonal to every function we can represent with our basis. The residual has "zero overlap" with our approximation space: we cannot express any part of it using our basis functions. In this sense, the residual is as "invisible" to our approximation as possible.

This condition is the defining property of optimality. By choosing our approximation $\hat{f}$ so that the residual $R = \mathscr{N}(\hat{f})$ is orthogonal to the entire approximation space, we ensure that $\hat{f}$ is the orthogonal projection of the true solution onto $\text{span}{\varphi_1, \ldots, \varphi_n}$. Within this $n$-dimensional space, no better choice is possible: any other coefficients would yield a residual with a nonzero component inside the space, and therefore a larger norm.

The finite-dimensional analogy makes this concrete. Suppose you want to approximate a vector $\mathbf{v} \in \mathbb{R}^3$ using only the $xy$-plane (a 2D subspace). The best approximation is to project $\mathbf{v}$ onto the plane, giving $\hat{\mathbf{v}} = (v_1, v_2, 0)$. The error is $\mathbf{r} = \mathbf{v} - \hat{\mathbf{v}} = (0, 0, v_3)$, which points purely in the $z$-direction—orthogonal to the entire $xy$-plane. This is precisely the Galerkin condition in action: the error is orthogonal to the approximation space.

#### Method of Moments: Test Against Monomials

The method of moments, for problems on $D \subset \mathbb{R}$, chooses test functions $p_i(x) = x^{i-1}$ for $i = 1, \ldots, n$:

$$
\langle R(\cdot; a), x^{i-1} \rangle = 0, \quad i = 1, \ldots, n.
$$

This requires the first $n$ moments of the residual function to vanish, ensuring the residual is "balanced" in the sense that it has no systematic trend captured by low-order polynomials. The moments $\int x^k R(x; a) w(x) dx$ measure weighted averages of the residual, with increasing powers of $x$ giving more weight to larger values. Setting these to zero ensures the residual doesn't grow systematically with $x$. This approach is particularly useful when $w(x)$ is chosen as a probability measure, making the conditions natural moment restrictions familiar from statistics and econometrics.

#### Collocation Method: Test Against Delta Functions

The collocation method chooses test functions $p_i(x) = \delta(x - x_i)$, the Dirac delta functions at points $\{x_1, \ldots, x_n\}$:

$$
\langle R(\cdot; a), \delta(\cdot - x_i) \rangle = R(x_i; a) = 0, \quad i = 1, \ldots, n.
$$

This is projection against the most localized test functions possible—delta functions that "sample" the residual at specific points, requiring the residual to vanish exactly where we test it. When using orthogonal polynomials with collocation points at the zeros of the $n$-th polynomial, the Chebyshev interpolation theorem guarantees that forcing $R(x_i; a) = 0$ at these specific points makes $R(x; a)$ small everywhere. The choice of collocation points is crucial: using the zeros of orthogonal polynomials produces well-conditioned systems and near-optimal interpolation error. The computational advantage is significant—collocation avoids numerical integration entirely, requiring only pointwise evaluation of $R$.

#### Subdomain Method: Test Against Indicator Functions

The subdomain method partitions the domain into $n$ subregions $\{D_1, \ldots, D_n\}$ and chooses test functions $p_i = I_{D_i}$, the indicator functions:

$$
\langle R(\cdot; a), I_{D_i} \rangle = \int_{D_i} R(x; a) w(x) dx = 0, \quad i = 1, \ldots, n.
$$

This requires the residual to have zero average over each subdomain, ensuring the approximation is good "on average" over each piece of the domain. This approach is particularly natural for finite element methods where the domain is divided into elements, ensuring local balance of the residual within each element.

#### Least Squares: An Alternative Framework

The least squares approach doesn't fit the test function framework directly. Instead, we minimize:

$$
\min_a \int_{\mathcal{S}} R(x; a)^2 w(x) dx = \min_a \langle R(\cdot; a), R(\cdot; a) \rangle.
$$

The first-order conditions for this minimization problem are:

$$
\left\langle R(\cdot; a), \frac{\partial R(\cdot; a)}{\partial a_i} \right\rangle = 0, \quad i = 1, \ldots, n.
$$

Thus least squares implicitly uses test functions $p_i = \partial R / \partial a_i$—the gradients of the residual with respect to parameters. Unlike other methods where test functions are chosen a priori, here they depend on the current guess for $a$ and on the structure of our approximation.

We can now see the unifying structure: all projection methods (except least squares in its direct form) follow the same template of picking $n$ test functions and requiring $\langle R, p_i \rangle = 0$. They differ only in their philosophy about which test functions best reveal whether the residual is "nearly zero." Galerkin tests against the approximation basis itself (natural for orthogonal bases), the method of moments tests against monomials (ensuring polynomial balance), collocation tests against delta functions (pointwise satisfaction), subdomain tests against indicators (local average satisfaction), and least squares tests against residual gradients (global norm minimization). Each choice reflects different priorities: computational efficiency, theoretical optimality, ease of implementation, or sensitivity to errors in different regions of the domain.

### Step 4: Solve the Finite-Dimensional Problem

The projection conditions give us a system to solve for the coefficients $a$. For test function methods (Galerkin, collocation, moments, subdomain), we solve:

$$
P_i(a) \equiv \langle R(\cdot; a), p_i \rangle = 0, \quad i = 1, \ldots, n.
$$

This is a system of $n$ (generally nonlinear) equations in $n$ unknowns. For least squares, we solve the optimization problem $\min_a \langle R(\cdot; a), R(\cdot; a) \rangle$.

**Computational characteristics**:

The **conditioning** of the system depends critically on the choice of test functions. The Jacobian matrix has entries:

$$
J_{ij} = \frac{\partial P_i}{\partial a_j} = \left\langle \frac{\partial R(\cdot; a)}{\partial a_j}, p_i \right\rangle.
$$

When test functions are orthogonal (or nearly so), the Jacobian tends to be well-conditioned. This is why orthogonal polynomial bases are preferred in Galerkin methods—they produce Jacobians with controlled condition numbers.

The **computational cost per iteration** varies significantly:
- **Collocation**: Cheapest to evaluate since $P_i(a) = R(x_i; a)$ requires only pointwise evaluation—no integration. The Jacobian is also cheap: $J_{ij} = \frac{\partial R(x_i; a)}{\partial a_j}$.
- **Galerkin and moments**: More expensive due to integration. Computing $P_i(a) = \int R(x; a) p_i(x) w(x) dx$ requires numerical quadrature. Each Jacobian entry requires integrating $\frac{\partial R}{\partial a_j} p_i$.
- **Least squares**: Most expensive when done via the objective function, which requires integrating $R^2$. However, the first-order conditions reduce it to a system like Galerkin, with test functions $p_i = \partial R / \partial a_i$.

**Interaction with quadrature**: For methods requiring integration, the choice of quadrature rule should match the basis. Gaussian quadrature with nodes at orthogonal polynomial zeros is particularly efficient—and when combined with collocation at those same points, the quadrature is exact for polynomials up to a certain degree. This coordination between quadrature and collocation is what makes **orthogonal collocation** particularly powerful.

The choice of solver depends critically on whether the finite-dimensional approximation preserves the structural properties of the original infinite-dimensional problem. This is particularly important for the Bellman equation, where the original operator $\mathrm{L}$ is a contraction.

**Successive approximation** (fixed-point iteration) is the natural choice when the original operator is a contraction, as it preserves the global convergence guarantees. However, the finite-dimensional approximation $\hat{\mathrm{L}}$ may not inherit the contraction property of $\mathrm{L}$. The approximation can introduce spurious fixed points or destroy the contraction constant, leading to divergence or slow convergence. This is especially problematic when using high-order polynomial approximations, which can create artificial oscillations that destabilize the iteration.

**Newton's method** is often the default choice for projection methods because it doesn't rely on the contraction property. Instead, it exploits the smoothness of the residual function. When the original problem is smooth and the approximation preserves this smoothness, Newton's method provides quadratic convergence near the solution. However, Newton's method requires good initial guesses and may converge to spurious solutions if the finite-dimensional problem has multiple fixed points that the original problem lacks.

**The choice of basis and projection method affects which algorithm is most appropriate**. For example:
- **Linear interpolation** often preserves contraction properties, making successive approximation reliable
- **High-order polynomials** may destroy contraction but provide smooth approximations suitable for Newton's method
- **Shape-preserving splines** can maintain both smoothness and structural properties

**In practice, which algorithm should we use?** When the operator equation can be written as a fixed-point problem $f = \mathscr{T}f$ and the operator $\mathscr{T}$ is known to be a contraction, successive approximation is often the best starting point—it is computationally cheap and globally convergent. However, not all equations $\mathscr{N}(f) = 0$ admit a natural fixed-point reformulation, and even when they do (e.g., $f = f - \alpha \mathscr{N}(f)$ for some $\alpha > 0$), the resulting operator may not be a contraction in the finite-dimensional approximation space. In such cases, Newton's method becomes the primary option despite its requirement for good initial guesses and higher computational cost per iteration. A hybrid approach often works well: use successive approximation when applicable to generate an initial guess, then switch to Newton's method for refinement.

An major consideration is the conditioning of the resulting system. Poorly chosen basis functions or collocation points can lead to nearly singular Jacobians, causing numerical instability. This is why orthogonal bases and carefully chosen collocation points (like Chebyshev nodes) are preferred—they tend to produce well-conditioned systems.

### Step 5: Verify the Solution

Once we have computed a candidate solution $\hat{f}$, we must verify its quality. Projection methods optimize $\hat{f}$ with respect to specific criteria (specific test functions or collocation points), but we should check that the residual is small everywhere, not just in the directions or at the points we optimized over.

Typical diagnostic checks include:
- Computing $\|R(\cdot; a)\|$ using a more accurate quadrature rule than was used in the optimization
- Evaluating $R(x; a)$ at many points not used in the fitting process
- If using Galerkin with the first $n$ basis functions, checking orthogonality against higher-order basis functions

## Application to the Bellman Equation

We now apply the projection method framework to the Bellman optimality equation. Recall that we seek a function $v$ satisfying

$$
v(s) = \mathrm{L}v(s) = \max_{a \in \mathcal{A}_s} \left\{ r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j) \right\}.
$$

Writing this as an operator equation $\mathscr{N}(v) = 0$ with $\mathscr{N}(v) = \mathrm{L}v - v$, the residual function for a candidate approximation $\hat{v}(s) = \sum_{i=1}^n a_i \varphi_i(s)$ is:

$$
R(s; a) = \mathrm{L}\hat{v}(s) - \hat{v}(s) = \max_{a \in \mathcal{A}_s} \left\{ r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) \hat{v}(j) \right\} - \sum_{i=1}^n a_i \varphi_i(s).
$$

Any of the projection methods we discussed—Galerkin, method of moments, collocation, subdomain, or least squares—can be applied here. Each would give us $n$ conditions to determine the $n$ coefficients in our approximation. For instance:
- **Galerkin** would require $\langle R(\cdot; a), \varphi_i \rangle = 0$ for $i = 1, \ldots, n$, involving integration of the residual weighted by basis functions
- **Method of moments** would require $\langle R(\cdot; a), s^{i-1} \rangle = 0$, setting the first $n$ moments of the residual to zero
- **Collocation** would require $R(s_i; a) = 0$ at $n$ chosen states, forcing the residual to vanish pointwise

In practice, **collocation is the most commonly used** projection method for the Bellman equation. The reason is computational: collocation avoids the numerical integration required by Galerkin and method of moments. Since the Bellman operator already involves integration (or summation) over next states, adding another layer of integration for the projection conditions would be computationally expensive. Collocation sidesteps this by requiring the equation to hold exactly at specific points.

We focus on collocation in detail, though the principles extend to other projection methods.

### Collocation for the Bellman Equation

The collocation approach chooses $n$ states $\{s_1, \ldots, s_n\}$ (the collocation points) and requires:

$$
R(s_i; a) = 0, \quad i = 1, \ldots, n.
$$

This gives us a system of $n$ nonlinear equations in $n$ unknowns:

$$
\sum_{j=1}^n a_j \varphi_j(s_i) = \max_{a \in \mathcal{A}_{s_i}} \left\{ r(s_i,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s_i,a) \hat{v}(j) \right\}, \quad i = 1, \ldots, n.
$$

The right-hand side requires evaluating the Bellman operator at the collocation points. For each collocation point $s_i$, we must:
1. For each action $a \in \mathcal{A}_{s_i}$, compute the expected continuation value $\sum_{j \in \mathcal{S}} p(j|s_i,a) \hat{v}(j)$
2. Take the maximum over actions

When the state space is continuous, the expectation involves integration, which typically requires numerical quadrature. When the state space is discrete but large, this is a straightforward (though potentially expensive) summation.

#### Solving the Collocation System

The resulting system is nonlinear due to the max operator on the right-hand side. We can write this more compactly as finding $a \in \mathbb{R}^n$ such that $F(a) = 0$ where:

$$
F_i(a) = \sum_{j=1}^n a_j \varphi_j(s_i) - \max_{u \in \mathcal{A}_{s_i}} \left\{ r(s_i,u) + \gamma \sum_{j \in \mathcal{S}} p(j|s_i,u) \hat{v}(j; a) \right\}.
$$

**Newton's method** is the standard approach for such systems. However, the max operator introduces a non-differentiability issue: the function $F(a)$ is not everywhere differentiable because the optimal action can change discontinuously as $a$ varies. Fortunately, the function is **semismooth**—it is locally Lipschitz continuous and directionally differentiable everywhere. This structure can be exploited by **semi-smooth Newton methods**, which generalize Newton's method to semismooth equations by using any element of the generalized Jacobian (in the sense of Clarke's generalized derivative) in place of the classical Jacobian.

In practice, implementing semi-smooth Newton for the Bellman equation is straightforward: at each iteration, we fix the optimal action $a^*(s_i; a^{(k)})$ at the current guess $a^{(k)}$, compute the Jacobian assuming these actions remain optimal, and update:

$$
a^{(k+1)} = a^{(k)} - J_F(a^{(k)})^{-1} F(a^{(k)}).
$$

As we approach the solution, the optimal actions typically stabilize, and the method achieves superlinear convergence despite the non-smoothness. The main practical requirement is a good initial guess, which can be obtained from the successive approximation method described next.

### Iterative Solution: Successive Approximation

Rather than solving the nonlinear system directly via Newton's method, we can exploit the fixed-point structure of the problem. The collocation equations can be viewed as requiring that the approximation matches the Bellman operator at the collocation points. This suggests an iterative scheme that performs **successive approximation** (fixed-point iteration) in the finite-dimensional coefficient space. Starting with an initial guess $a^{(0)}$, we iterate:

1. **Maximization step**: At each collocation point $s_i$, compute

   $$
   v_i^{(k+1)} = \max_{a \in \mathcal{A}_{s_i}} \left\{ r(s_i,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s_i,a) \hat{v}(s_j; a^{(k)}) \right\}.
   $$

2. **Fitting step**: Find coefficients $a^{(k+1)}$ such that $\hat{v}(s_i; a^{(k+1)}) = v_i^{(k+1)}$ for all $i = 1, \ldots, n$. This is a linear system if our approximation is linear in the coefficients.

3. **Check convergence**: If $\|a^{(k+1)} - a^{(k)}\|$ is sufficiently small, stop; otherwise return to step 1.

This algorithm, which Judd calls **parametric value function iteration** or **projection-based value iteration**, separates the difficult nonlinear optimization (the max operator in the Bellman equation) from the approximation problem. Each iteration improves the approximation by ensuring it matches the Bellman operator at the collocation points. Mathematically, it performs successive approximation in the finite-dimensional coefficient space: we define an operator $\hat{\mathrm{L}}$ that maps coefficients $a^{(k)}$ to new coefficients $a^{(k+1)}$ via the maximization and fitting steps, then iterate $a^{(k+1)} = \hat{\mathrm{L}}(a^{(k)})$.

#### Comparison of Solution Methods

We now have two approaches to solving the collocation equations:

1. **Semi-smooth Newton**: Solve the nonlinear system $F(a) = 0$ directly using Newton's method adapted for semismooth functions. This offers fast (superlinear) convergence near the solution but requires a good initial guess and may fail to converge from poor starting points.

2. **Successive approximation (parametric value iteration)**: Iterate the map $a^{(k+1)} = \hat{\mathrm{L}}(a^{(k)})$ that alternates between maximization and fitting steps. This is more robust to poor initial guesses and inherits global convergence properties when the finite-dimensional approximation preserves the contraction property of the Bellman operator.

**Which should we use?** Following Judd's guidance: When the Bellman operator is known to be a contraction and the finite-dimensional approximation preserves this property (as with linear interpolation or carefully chosen low-order approximations), successive approximation is often the best choice—it is globally convergent and each iteration is relatively cheap. However, high-order polynomial approximations may destroy the contraction property or introduce numerical instabilities. In such cases, or when convergence is too slow, Newton's method (or semi-smooth Newton) becomes necessary despite requiring good initial guesses.

A **hybrid strategy** works well in practice: use successive approximation to generate an initial approximation, then switch to semi-smooth Newton for rapid refinement once in the neighborhood of the solution. This combines the global convergence of successive approximation with the fast local convergence of Newton's method.

### Shape-Preserving Considerations

A subtle but important issue arises in dynamic programming: the value function typically has specific structural properties that we want our approximation to preserve. For instance:
- **Monotonicity**: If having more of a resource is better, the value function should be increasing
- **Concavity**: Diminishing returns often imply concave value functions
- **Boundedness**: The value function is bounded when rewards are bounded

Standard polynomial approximation does not automatically preserve these properties. A polynomial fit to increasing, concave data points can produce a function with non-monotonic or convex regions between the data points. This can destabilize the iterative algorithm: artificially high values at non-collocation points can lead to poor decisions in the maximization step, which feeds back into even worse approximations.

**Shape-preserving approximation methods** address this issue. For one-dimensional problems, Schumaker's shape-preserving quadratic splines maintain monotonicity and concavity while providing continuously differentiable approximations. For multidimensional problems, linear interpolation on simplices preserves monotonicity and convex combinations (though not concavity or smoothness).

The trade-off is between smoothness and shape preservation. Smooth approximations (high-order polynomials or splines) enable efficient optimization in the maximization step through gradient-based methods, but risk introducing spurious features. Simple approximations (linear interpolation) guarantee shape preservation but introduce kinks that complicate optimization and may produce discontinuous policies when the true policy is continuous.

## Galerkin Projection and Least Squares Temporal Difference

An important special case emerges when we apply Galerkin projection to the **policy evaluation** problem rather than the optimality problem. For a fixed policy $\pi$, the policy evaluation Bellman equation is:

$$
v^\pi(s) = \mathrm{L}_\pi v^\pi(s) = r(s,\pi(s)) + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,\pi(s)) v^\pi(s').
$$

This is a linear operator (no max), making the projection problem significantly simpler. Consider a linear function approximation $\hat{v}(s) = \boldsymbol{\varphi}(s)^\top \mathbf{a}$ where $\boldsymbol{\varphi}(s) = [\varphi_1(s), \ldots, \varphi_n(s)]^\top$ are basis functions and $\mathbf{a} = [a_1, \ldots, a_n]^\top$ are coefficients to determine. The residual is:

$$
R(s; \mathbf{a}) = \mathrm{L}_\pi \hat{v}(s) - \hat{v}(s) = r(s,\pi(s)) + \gamma \sum_{s'} p(s'|s,\pi(s)) \boldsymbol{\varphi}(s')^\top \mathbf{a} - \boldsymbol{\varphi}(s)^\top \mathbf{a}.
$$

The Galerkin projection requires the residual to be orthogonal to all basis functions with respect to some weighting:

$$
\sum_{s \in \mathcal{S}} d(s) R(s; \mathbf{a}) \varphi_j(s) = 0, \quad j = 1, \ldots, n,
$$

where $d(s)$ is a distribution over states (often the stationary distribution under policy $\pi$, or uniform over visited states). Substituting the residual:

$$
\sum_s d(s) \left[ r(s,\pi(s)) + \gamma \sum_{s'} p(s'|s,\pi(s)) \boldsymbol{\varphi}(s')^\top \mathbf{a} - \boldsymbol{\varphi}(s)^\top \mathbf{a} \right] \varphi_j(s) = 0.
$$

Rearranging and writing in matrix form, let $\mathbf{D}$ be a diagonal matrix with $D_{ss} = d(s)$, $\boldsymbol{\Phi}$ be the $|\mathcal{S}| \times n$ matrix with rows $\boldsymbol{\varphi}(s)^\top$, and $\mathbf{P}_\pi$ be the transition matrix under policy $\pi$. The Galerkin conditions become:

$$
\boldsymbol{\Phi}^\top \mathbf{D} (\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \boldsymbol{\Phi} \mathbf{a} - \boldsymbol{\Phi} \mathbf{a}) = \mathbf{0}.
$$

Solving for $\mathbf{a}$:

$$
\boldsymbol{\Phi}^\top \mathbf{D} (\boldsymbol{\Phi} - \gamma \mathbf{P}_\pi \boldsymbol{\Phi}) \mathbf{a} = \boldsymbol{\Phi}^\top \mathbf{D} \mathbf{r}_\pi.
$$

This is precisely the **Least Squares Temporal Difference (LSTD)** solution for policy evaluation. The connection reveals that LSTD is Galerkin projection applied to the linear policy evaluation Bellman equation. The "least squares" name comes from the fact that this is the projection (in the weighted $\ell^2$ sense) of the Bellman operator's output onto the span of the basis functions.

Viewing LSTD through the projection lens reveals something fundamental about approximate dynamic programming. The solution $\mathbf{a}$ does not satisfy the true Bellman equation $v = \mathrm{L}_\pi v$—which is typically impossible within our finite-dimensional approximation space. Instead, it satisfies $\hat{v} = \Pi \mathrm{L}_\pi \hat{v}$, where $\Pi$ is the projection operator onto $\text{span}\{\varphi_1, \ldots, \varphi_n\}$. We find the fixed point of the *projected* Bellman operator, not the Bellman operator itself. This is why approximation error persists even at convergence: the best we can do is find the value function whose Bellman operator output projects back onto itself.

This operator composition $\Pi \mathrm{L}_\pi$ has important contraction properties. The Bellman operator $\mathrm{L}_\pi$ is a $\gamma$-contraction in any weighted $\ell^\infty$ norm, while the projection operator $\Pi$ (with respect to the $d$-weighted $\ell^2$ norm) is a non-expansion: it cannot increase distances. However, the composition $\Pi \mathrm{L}_\pi$ is generally *not* a contraction in the $\ell^2$ norm—projecting after contracting can amplify certain components of the error.

The exception is the **on-policy case**: when the weighting distribution $d(s)$ in the projection matches the stationary distribution of policy $\pi$. In this setting, $\Pi \mathrm{L}_\pi$ becomes a $\gamma$-contraction in the $d$-weighted $\ell^2$ norm. This guarantees that iterative methods for solving the LSTD equations (such as LSTDQ or temporal difference learning with linear function approximation) will converge to the unique fixed point. The on-policy condition ensures that the projection and the dynamics "align"—we're projecting with respect to the same distribution that governs how the Bellman operator propagates values through state transitions.

When $d$ differs from the stationary distribution (the off-policy case), convergence is not guaranteed, and additional techniques like importance sampling corrections become necessary to restore stability.

The linearity of the policy evaluation operator $\mathrm{L}_\pi$ is what gives us the closed-form solution. We could apply Galerkin projection to the Bellman optimality equation $v^* = \mathrm{L} v^*$, setting up orthogonality conditions $\sum_s d(s) R(s; \mathbf{a}) \varphi_j(s) = 0$. The max operator makes these conditions nonlinear in $\mathbf{a}$, eliminating the closed form and requiring iterative solution—which brings us back to the successive approximation methods discussed earlier for collocation.

This framework of projection methods—choosing test functions, defining residuals, and solving finite-dimensional systems—provides the conceptual foundation for approximate dynamic programming. However, we've left one critical question unresolved: how do we evaluate the expectations in the Bellman operator when we lack explicit transition probabilities or when the state space is too large for exact computation? The next chapter addresses this by introducing Monte Carlo integration methods, completing the bridge from classical projection methods to modern simulation-based approximate dynamic programming and reinforcement learning.



