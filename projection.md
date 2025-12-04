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


# Weighted Residual Methods for Functional Equations

The Bellman optimality equation $\Bellman v = v$ is a functional equation: an equation where the unknown is an entire function rather than a finite-dimensional vector. When the state space is continuous or very large, we cannot represent the value function exactly on a computer. We must instead work with finite-dimensional approximations. This motivates weighted residual methods (also called minimum residual methods), a general framework for transforming infinite-dimensional problems into tractable finite-dimensional ones {cite}`Chakraverty2019,AtkinsonPotra1987`.

## Testing Whether a Residual Vanishes

Consider a functional equation $\Residual(f) = 0$, where $\Residual$ is an operator and the unknown $f$ is an entire function (in our case, the Bellman optimality equation $\Bellman v = v$, which we can write as $\Residual(v) \equiv \Bellman v - v = 0$). Suppose we have found a candidate approximate solution $\hat{f}$. To verify it satisfies $\Residual(\hat{f}) = 0$, we compute the **residual function** $R(s) = \Residual(\hat{f})(s)$. For a true solution, this residual should be the **zero function**: $R(s) = 0$ for every state $s$. 

How might we test whether a function is zero? One approach: sample many input points $\{s_1, s_2, \ldots, s_m\}$, check whether $R(s_i) = 0$ at each, and summarize the results into a single scalar test by computing a weighted sum $\sum_{i=1}^m w_i R(s_i)$ with weights $w_i > 0$. If $R$ is zero everywhere, this sum is zero. If $R$ is nonzero somewhere, we can choose points and weights to make the sum nonzero. For vectors in finite dimensions, the inner product $\langle \mathbf{r}, \mathbf{y} \rangle = \sum_{i=1}^n r_i y_i$ implements exactly this idea: it tests $\mathbf{r}$ by weighting and summing. Indeed, a vector $\mathbf{r} \in \mathbb{R}^n$ equals zero if and only if $\langle \mathbf{r}, \mathbf{y} \rangle = 0$ for every vector $\mathbf{y} \in \mathbb{R}^n$. To see why, suppose $\mathbf{r} \neq \mathbf{0}$. Choosing $\mathbf{y} = \mathbf{r}$ gives $\langle \mathbf{r}, \mathbf{r} \rangle = \|\mathbf{r}\|^2 > 0$, contradicting the claim that all inner products vanish.

The same principle extends to functions. A function $R$ equals the zero function if and only if its "inner product" with every "test function" $p$ vanishes:

$$
R = 0 \quad \text{if and only if} \quad \langle R, p \rangle_w = \int_{\mathcal{S}} R(s) p(s) w(s) ds = 0 \quad \text{for all test functions } p,
$$

where $w(s) > 0$ is a weight function that is part of the inner product definition. Why does this work? For the same reason as in finite dimensions: if $R$ is not the zero function, there must be some region where $R(s) \neq 0$. We can then choose a test function $p$ that is nonzero in that same region (for instance, $p(s) = R(s)$ itself), which will produce $\langle R, p \rangle_w = \int R(s) p(s) w(s) ds > 0$, witnessing that $R$ is nonzero. Conversely, if $R$ is the zero function, then $\langle R, p \rangle_w = 0$ for any test function $p$.

This ability to **distinguish between different functions using inner products** is a fundamental principle from functional analysis. Just as we can test a vector by taking inner products with other vectors, we can test a function by taking inner products with other functions.

```{admonition} Connection to Functional Analysis
:class: dropdown

The principle that "a function equals zero if and only if it has zero inner product with all test functions" is a consequence of the **Hahn-Banach theorem**, one of the cornerstones of functional analysis. The theorem guarantees that for any nonzero function $R$ in a suitable function space, there exists a continuous linear functional (which can be represented as an inner product with some test function $p$) that produces a nonzero value when applied to $R$. This is often phrased as "the dual space separates points."

While you don't need to know the Hahn-Banach theorem to use weighted residual methods, it provides the rigorous mathematical foundation ensuring that our inner product tests are theoretically sound. The constructive argument we gave above (choosing $p = R$) works in simple cases with well-behaved functions, but the Hahn-Banach theorem extends this guarantee to much more general settings.
```

Why is this useful? It transforms the pointwise condition "$R(s) = 0$ for all $s$" (infinitely many conditions, one per state) into an equivalent condition about inner products. We still cannot test against *all* possible test functions, since there are infinitely many of those too. But the inner product perspective suggests a natural computational strategy: choose a finite collection of test functions $\{p_1, \ldots, p_n\}$ and use them to construct $n$ conditions that we can actually compute.

This suggests the **weighted residual** (or **minimum residual**) approach: choose $n$ test functions $\{p_1, \ldots, p_n\}$ and a weight function $w(s)$, then require the residual to have zero weighted inner product with each test function:

$$
\langle R, p_i \rangle_w = \int_{\mathcal{S}} R(s; \theta) p_i(s) w(s) ds = 0, \quad i = 1, \ldots, n,
$$

where the residual is $R(s; \theta) = \Residual(\hat{f}(\cdot; \theta))(s)$ and the approximation is $\hat{f}(s; \theta) = \sum_{j=1}^n \theta_j \varphi_j(s)$. For the Bellman equation, $\Residual(v) = \Bellman v - v$, so $R(s; \theta) = \Bellman\hat{v}(s; \theta) - \hat{v}(s; \theta)$. This transforms the impossible task of verifying "$R(s) = 0$ for all $s$" into a finite-dimensional problem: find $n$ coefficients $\theta = (\theta_1, \ldots, \theta_n)$ satisfying $n$ weighted integral conditions.

The weight function $w(s)$ is part of the inner product definition and serves important purposes: it can emphasize certain regions of the state space, or represent a natural probability measure over states. In unweighted problems, we simply take $w(s) = 1$. In reinforcement learning applications, $w(s)$ is often chosen as the stationary distribution $d^\pi(s)$ under a policy $\pi$. This is exactly what happens in methods like LSTD (Least Squares Temporal Difference), which can be viewed as a Galerkin method with $w(s) = d^\pi(s)$.

Different choices of test functions give different methods, each with different computational and theoretical properties. When $p_i = \varphi_i$ (test functions equal basis functions), the method is called **Galerkin**, and can be interpreted geometrically as a **projection** of the residual onto the orthogonal complement of the approximation space. The rest of this chapter develops this framework systematically: how to choose basis functions $\{\varphi_j\}$, how to select test functions $\{p_i\}$, and how to solve the resulting systems. 

## The General Framework

Consider an operator equation of the form

$$
\Residual(f) = 0,
$$

where $\Residual: B_1 \to B_2$ is a continuous operator between complete normed vector spaces $B_1$ and $B_2$. For the Bellman equation, we have $\Residual(v) = \Bellman v - v$, so that solving $\Residual(v) = 0$ is equivalent to finding the fixed point $v = \Bellman v$.

Just as we transcribed infinite-dimensional continuous optimal control problems into finite-dimensional discrete optimal control problems in earlier chapters, we seek a finite-dimensional approximation to this infinite-dimensional functional equation. Recall that for continuous optimal control, we adopted control parameterization: we represented the control trajectory using a finite set of basis functions (piecewise constants, polynomials, splines) and searched over the finite-dimensional coefficient space instead of the infinite-dimensional function space. For integrals in the objective and constraints, we used numerical quadrature to approximate them with finite sums.

We follow the same strategy here. We parameterize the value function using a finite set of basis functions $\{\varphi_1, \ldots, \varphi_n\}$, commonly polynomials (Chebyshev, Legendre), though other function classes (splines, radial basis functions, neural networks) are possible, and search for coefficients $\theta = (\theta_1, \ldots, \theta_n)$ in $\mathbb{R}^n$. When integrals appear in the Bellman operator or projection conditions, we approximate them using numerical quadrature. The projection method approach consists of several conceptual steps that accomplish this transcription.

### Step 1: Choose a Finite-Dimensional Approximation Space

We begin by selecting a basis $\Phi = \{\varphi_1, \varphi_2, \ldots, \varphi_n\}$ and approximating the unknown function as a linear combination:

$$
\hat{f}(x) = \sum_{i=1}^n \theta_i \varphi_i(x).
$$

The choice of basis functions $\varphi_i$ is problem-dependent. Common choices include:
- **Polynomials**: For smooth problems, we might use Chebyshev polynomials or other orthogonal polynomial families
- **Splines**: For problems where we expect the solution to have regions of different smoothness
- **Radial basis functions**: For high-dimensional problems where tensor product methods become intractable

The number of basis functions $n$ determines the flexibility of our approximation. In practice, we start with small $n$ and increase it until the approximation quality is satisfactory. The only unknowns now are the coefficients $\theta = (\theta_1, \ldots, \theta_n)$.

While the classical presentation of projection methods focuses on polynomial bases, the framework applies equally well to other function classes. Neural networks, for instance, can be viewed through this lens: a neural network $\hat{f}(x; \theta)$ with parameters $\theta$ defines a flexible function class, and many training procedures can be interpreted as projection methods with specific choices of test functions or residual norms. The distinction is that classical methods typically use predetermined basis functions with linear coefficients, while neural networks use adaptive nonlinear features. Throughout this chapter, we focus on the classical setting to develop the core concepts, but the principles extend naturally to modern function approximators.


### Step 2: Define the Residual Function

Since we are approximating $f$ with $\hat{f}$, the operator $\Residual$ will generally not vanish exactly. Instead, we obtain a **residual function**:

$$
R(x; \theta) = \Residual(\hat{f}(\cdot; \theta))(x).
$$

This residual measures how far our candidate solution is from satisfying the equation at each point $x$. As we discussed in the introduction, we will assess whether this residual is "close to zero" by testing its inner products against chosen test functions.

### Step 3: Impose Conditions to Determine Coefficients

Having chosen our basis and defined the residual, we must decide how to make the residual "close to zero." As discussed in the introduction, we test the residual using **weighted integrals**. We select test functions $\{p_1, \ldots, p_n\}$ and a weight function $w(x)$ (often $w(x) = 1$ for unweighted problems), then impose **weighted residual conditions**:

$$
\int_{\mathcal{S}} R(x; \theta) p_i(x) w(x) \, dx = 0, \quad i = 1, \ldots, n.
$$

These $n$ integral conditions provide $n$ equations to determine the $n$ unknown coefficients in $\theta$. Different choices of test functions $p_i$ give different methods:
- **Galerkin**: $p_i = \varphi_i$ (test with the basis functions themselves)
- **Collocation**: $p_i = \delta(x - x_i)$ (test at specific points)
- **Method of moments**: $p_i = x^{i-1}$ (test with monomials)

An alternative is the **least squares approach**, which minimizes the weighted norm of the residual:

$$
\min_\theta \int_{\mathcal{S}} R(x; \theta)^2 w(x) dx.
$$

We focus primarily on methods distinguished by their choice of test functions $p_i$. The Galerkin method, where $p_i = \varphi_i$, can be interpreted geometrically as a **projection** when working in a Hilbert space with a weighted inner product.

Let us examine the standard choices of test functions and what they tell us about the residual:

#### Galerkin Method: Test Against the Basis

The Galerkin method chooses test functions $p_i = \varphi_i$, the same basis functions used to approximate $\hat{f}$:

$$
\int_{\mathcal{S}} R(x; \theta) \varphi_i(x) w(x) dx = 0, \quad i = 1, \ldots, n.
$$

To understand what this means, recall that in finite dimensions, two vectors are orthogonal when their inner product is zero. For functions, $\langle R, \varphi_i \rangle_w = \int R(x) \varphi_i(x) w(x) dx = 0$ expresses the same concept: $R$ and $\varphi_i$ are orthogonal as functions with respect to the weighted inner product. But there's more to this than just testing against individual basis functions.

Consider our approximation space $\text{span}\{\varphi_1, \ldots, \varphi_n\}$ as an $n$-dimensional subspace within the infinite-dimensional space of all functions. Any function $g$ in this space can be written as $g = \sum_{i=1}^n c_i \varphi_i$ for some coefficients $c_i$. If the residual $R$ is orthogonal to all basis functions $\varphi_i$, then by linearity of the inner product, for any such function $g$:

$$
\langle R, g \rangle = \left\langle R, \sum_{i=1}^n c_i \varphi_i \right\rangle = \sum_{i=1}^n c_i \langle R, \varphi_i \rangle = 0.
$$

This shows that $R$ is orthogonal to every function we can represent with our basis. The residual has "zero overlap" with our approximation space: we cannot express any part of it using our basis functions. In this sense, the residual is as "invisible" to our approximation as possible.

This condition is the defining property of optimality. By choosing our approximation $\hat{f}$ so that the residual $R = \Residual(\hat{f})$ is orthogonal to the entire approximation space, we ensure that $\hat{f}$ is the orthogonal projection of the true solution onto $\text{span}{\varphi_1, \ldots, \varphi_n}$. Within this $n$-dimensional space, no better choice is possible: any other coefficients would yield a residual with a nonzero component inside the space, and therefore a larger norm.

The finite-dimensional analogy makes this concrete. Suppose you want to approximate a vector $\mathbf{v} \in \mathbb{R}^3$ using only the $xy$-plane (a 2D subspace). The best approximation is to project $\mathbf{v}$ onto the plane, giving $\hat{\mathbf{v}} = (v_1, v_2, 0)$. The error is $\mathbf{r} = \mathbf{v} - \hat{\mathbf{v}} = (0, 0, v_3)$, which points purely in the $z$-direction, orthogonal to the entire $xy$-plane. We see the Galerkin condition in action: the error is orthogonal to the approximation space.

#### Method of Moments: Test Against Monomials

The method of moments, for problems on $D \subset \mathbb{R}$, chooses test functions $p_i(x) = x^{i-1}$ for $i = 1, \ldots, n$:

$$
\langle R(\cdot; \theta), x^{i-1} \rangle = 0, \quad i = 1, \ldots, n.
$$

This requires the first $n$ moments of the residual function to vanish, ensuring the residual is "balanced" in the sense that it has no systematic trend captured by low-order polynomials. The moments $\int x^k R(x; \theta) w(x) dx$ measure weighted averages of the residual, with increasing powers of $x$ giving more weight to larger values. Setting these to zero ensures the residual doesn't grow systematically with $x$. This approach is particularly useful when $w(x)$ is chosen as a probability measure, making the conditions natural moment restrictions familiar from statistics and econometrics.

#### Collocation Method: Test Against Delta Functions

The collocation method chooses test functions $p_i(x) = \delta(x - x_i)$, the Dirac delta functions at points $\{x_1, \ldots, x_n\}$:

$$
\langle R(\cdot; \theta), \delta(\cdot - x_i) \rangle = R(x_i; \theta) = 0, \quad i = 1, \ldots, n.
$$

This is projection against the most localized test functions possible: delta functions that "sample" the residual at specific points, requiring the residual to vanish exactly where we test it. The computational advantage is significant: collocation avoids numerical integration entirely, requiring only pointwise evaluation of $R$.

**Orthogonal collocation** combines collocation with spectral basis functions (orthogonal polynomials like Chebyshev, Legendre, or Hermite) for smooth problems. We choose collocation points at the **zeros of the $n$-th polynomial** in the family. For example, with Chebyshev polynomials $T_0, T_1, \ldots, T_{n-1}$, we place collocation points at the zeros of $T_n(x)$.

These zeros are also optimal nodes for **Gauss quadrature** with the associated weight function. This coordination means:
- We get the computational simplicity of collocation: just pointwise evaluation $R(x_i) = 0$
- When we need integrals (inside the Bellman operator), the collocation points double as quadrature nodes with exactness for polynomials up to degree $2n-1$
- For smooth problems, spectral approximations achieve **exponential convergence**: the error decreases like $O(e^{-cn})$ as we add basis functions, compared to $O(h^{p+1})$ for piecewise polynomials

This approach is often called a **pseudospectral method** or **spectral collocation method**. The Chebyshev interpolation theorem guarantees that forcing $R(x_i; \theta) = 0$ at these carefully chosen points makes $R(x; \theta)$ small everywhere, with well-conditioned systems and near-optimal interpolation error.

#### Subdomain Method: Test Against Indicator Functions

The subdomain method partitions the domain into $n$ subregions $\{D_1, \ldots, D_n\}$ and chooses test functions $p_i = I_{D_i}$, the indicator functions:

$$
\langle R(\cdot; \theta), I_{D_i} \rangle_w = \int_{D_i} R(x; \theta) w(x) dx = 0, \quad i = 1, \ldots, n.
$$

This requires the residual to have zero average over each subdomain, ensuring the approximation is good "on average" over each piece of the domain. This approach is particularly natural for finite element methods where the domain is divided into elements, ensuring local balance of the residual within each element.

#### Least Squares

The least squares approach appears different at first glance, but it also fits the test function framework. We minimize:

$$
\min_\theta \int_{\mathcal{S}} R(x; \theta)^2 w(x) dx = \min_\theta \langle R(\cdot; \theta), R(\cdot; \theta) \rangle_w.
$$

The first-order conditions for this minimization problem are:

$$
\left\langle R(\cdot; \theta), \frac{\partial R(\cdot; \theta)}{\partial \theta_i} \right\rangle_w = 0, \quad i = 1, \ldots, n.
$$

Thus least squares implicitly uses test functions $p_i = \partial R / \partial \theta_i$, the gradients of the residual with respect to parameters. Unlike other methods where test functions are chosen a priori, here they depend on the current guess for $\theta$ and on the structure of our approximation.

We can now see the unifying structure of **weighted residual methods**: whether we use projection conditions or least squares minimization, all these methods follow the same template of restricting the search to an $n$-dimensional function space and imposing $n$ conditions on the residual. For projection methods specifically, we pick $n$ test functions and require $\langle R, p_i \rangle = 0$. They differ only in their philosophy about which test functions best detect whether the residual is "nearly zero." Galerkin tests against the approximation basis itself (natural for orthogonal bases), the method of moments tests against monomials (ensuring polynomial balance), collocation tests against delta functions (pointwise satisfaction), subdomain tests against indicators (local average satisfaction), and least squares tests against residual gradients (global norm minimization). Each choice reflects different priorities: computational efficiency, theoretical optimality, ease of implementation, or sensitivity to errors in different regions of the domain.

### Step 4: Solve the Finite-Dimensional Problem

The projection conditions give us a system to solve for the coefficients $\theta$. For test function methods (Galerkin, collocation, moments, subdomain), we solve:

$$
P_i(\theta) \equiv \langle R(\cdot; \theta), p_i \rangle_w = 0, \quad i = 1, \ldots, n.
$$

This is a system of $n$ (generally nonlinear) equations in $n$ unknowns. For least squares, we solve the optimization problem $\min_\theta \langle R(\cdot; \theta), R(\cdot; \theta) \rangle_w$.

#### Computational Cost and Conditioning

The **computational cost per iteration** varies significantly across methods:
- **Collocation**: Cheapest to evaluate since $P_i(\theta) = R(x_i; \theta)$ requires only pointwise evaluation (no integration). The Jacobian is also cheap: $J_{ij} = \frac{\partial R(x_i; \theta)}{\partial \theta_j}$.
- **Galerkin and moments**: More expensive due to integration. Computing $P_i(\theta) = \int R(x; \theta) p_i(x) w(x) dx$ requires numerical quadrature. Each Jacobian entry requires integrating $\frac{\partial R}{\partial \theta_j} p_i w$.
- **Least squares**: Most expensive when done via the objective function, which requires integrating $R^2 w$. However, the first-order conditions reduce it to a system like Galerkin, with test functions $p_i = \partial R / \partial \theta_i$.

For methods requiring integration, the choice of quadrature rule should match the basis. Gaussian quadrature with nodes at orthogonal polynomial zeros is efficient. When combined with collocation at those same points, the quadrature is exact for polynomials up to a certain degree. This coordination between quadrature and collocation makes **orthogonal collocation** effective.

The **conditioning** of the system depends on the choice of test functions. The Jacobian matrix has entries:

$$
J_{ij} = \frac{\partial P_i}{\partial \theta_j} = \left\langle \frac{\partial R(\cdot; \theta)}{\partial \theta_j}, p_i \right\rangle_w.
$$

When test functions are orthogonal (or nearly so), the Jacobian tends to be well-conditioned. This is why orthogonal polynomial bases are preferred in Galerkin methods: they produce Jacobians with controlled condition numbers. Poorly chosen basis functions or collocation points can lead to nearly singular Jacobians, causing numerical instability. Orthogonal bases and carefully chosen collocation points (like Chebyshev nodes) help maintain good conditioning.

#### Two Main Solution Approaches

We have two fundamentally different ways to solve the projection equations: **function iteration** (exploiting fixed-point structure) and **Newton's method** (exploiting smoothness). The choice depends on whether the original operator equation has contraction properties and how well those properties are preserved by the finite-dimensional approximation.

##### Method 1: Function Iteration (Successive Approximation)

When the operator equation has the form $f = \Contraction f$ where $\Contraction$ is a contraction, the most natural approach is to iterate the operator directly:

$$
\hat{f}^{(k+1)} = \Contraction \hat{f}^{(k)}.
$$

The infinite-dimensional iteration becomes a finite-dimensional iteration in coefficient space once we choose our weighted residual method. Given a current approximation $\hat{f}^{(k)}(x; \theta^{(k)})$, how do we find the coefficients $\theta^{(k+1)}$ for the next iterate $\hat{f}^{(k+1)}$?

Different weighted residual methods answer this differently. For **collocation**, we proceed in two steps:

1. **Evaluate the operator**: At each collocation point $x_i$, compute what the next iterate should be: $t_i^{(k)} = (\Contraction \hat{f}^{(k)})(x_i)$. These $n$ target values tell us what $\hat{f}^{(k+1)}$ should equal at the collocation points.

2. **Find matching coefficients**: Determine $\theta^{(k+1)}$ so that $\hat{f}^{(k+1)}(x_i; \theta^{(k+1)}) = t_i^{(k)}$ for all $i$. This is a linear system: $\sum_j \theta_j^{(k+1)} \varphi_j(x_i) = t_i^{(k)}$.

In matrix form: $\boldsymbol{\Phi} \theta^{(k+1)} = t^{(k)}$, where $\boldsymbol{\Phi}$ is the collocation matrix with entries $\Phi_{ij} = \varphi_j(x_i)$. Solving this system gives $\theta^{(k+1)} = \boldsymbol{\Phi}^{-1} t^{(k)}$.

For **Galerkin**, the projection condition $\langle \hat{f}^{(k+1)} - \Contraction \hat{f}^{(k+1)}, \varphi_i \rangle_w = 0$ directly gives a system for $\theta^{(k+1)}$. When $\Contraction$ is linear in its argument (as in many integral equations), this is a linear system. When $\Contraction$ is nonlinear (as in the Bellman equation), we must solve a nonlinear system at each iteration, though each solution still only involves $n$ unknowns rather than an infinite-dimensional function.

When $\Contraction$ is a contraction in the infinite-dimensional space with constant $\gamma < 1$, iterating it pulls any starting function toward the unique fixed point. The hope is that the finite-dimensional operator, evaluating $\Contraction$ and projecting back onto the span of the basis functions, inherits this contraction property. When it does, function iteration converges globally from any initial guess, with each iteration reducing the error by a factor of roughly $\gamma$. This is computationally attractive: we only evaluate the operator and solve a linear system (for collocation) or a relatively simple system (for other methods).

However, the finite-dimensional approximation doesn't always preserve contraction. High-order polynomial bases, in particular, can create oscillations between basis functions that amplify rather than contract errors. Even when contraction is preserved, convergence can be painfully slow when $\gamma$ is close to 1, the "weak contraction" regime common in economic problems with patient agents ($\gamma \approx 0.95$ or higher). Finally, not all operator equations naturally present themselves as contractions; some require reformulation (like $f = f - \alpha \Residual(f)$), and finding a good $\alpha$ can be problem-specific.

##### Method 2: Newton's Method

Alternatively, we can treat the projection equations as a rootfinding problem $G(\theta) = 0$ where $G_i(\theta) = P_i(\theta)$ for test function methods, or solve the first-order conditions for least squares. **Newton's method** uses the update:

$$
\theta^{(k+1)} = \theta^{(k)} - J_G(\theta^{(k)})^{-1} G(\theta^{(k)}),
$$

where $J_G(\theta)$ is the Jacobian of $G$ at $\theta$.

**Computing the Jacobian:** We must compute $J_{ij} = \frac{\partial G_i}{\partial \theta_j}$. For collocation, $G_i(\theta) = \hat{f}(x_i; \theta) - (\Contraction \hat{f}(\cdot; \theta))(x_i)$, so:

$$
\frac{\partial G_i}{\partial \theta_j} = \frac{\partial \hat{f}(x_i; \theta)}{\partial \theta_j} - \frac{\partial (\Contraction \hat{f}(\cdot; \theta))(x_i)}{\partial \theta_j}.
$$

The first term is straightforward (it's just $\varphi_j(x_i)$ for a linear approximation). The second term requires differentiating the operator $\Contraction$ with respect to the parameters.

When $\Contraction$ involves optimization (as in the Bellman operator $\Bellman v = \max_a \{r(s,a) + \gamma \mathbb{E}[v(s')]\}$), computing this derivative appears problematic because the max operator is not differentiable. However, the **Envelope Theorem** resolves this difficulty.

```{admonition} The Envelope Theorem
:class: important

**Setup:** Consider a smooth objective function $f(\mathbf{x}, \boldsymbol{\theta})$ and define the optimal value:

$$
v(\boldsymbol{\theta}) = \max_{\mathbf{x}} f(\mathbf{x}, \boldsymbol{\theta}).
$$

Let $\mathbf{x}^*(\boldsymbol{\theta})$ denote the maximizer, satisfying the first-order condition $\nabla_{\mathbf{x}} f(\mathbf{x}^*(\boldsymbol{\theta}), \boldsymbol{\theta}) = \mathbf{0}$.

**The Result:** To find how the optimal value changes with $\boldsymbol{\theta}$, we can compute:

$$
\nabla_{\boldsymbol{\theta}} v(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} f(\mathbf{x}^*(\boldsymbol{\theta}), \boldsymbol{\theta}).
$$

That is, differentiate the objective with respect to $\boldsymbol{\theta}$ while treating the maximizer $\mathbf{x}^*$ as constant. We don't need to compute $\frac{\partial \mathbf{x}^*}{\partial \boldsymbol{\theta}}$ because at the optimum, small changes in $\mathbf{x}$ don't affect the value (first-order condition), so the direct effect dominates.

**Why it works:** By the chain rule, $\nabla_{\boldsymbol{\theta}} v = \nabla_{\boldsymbol{\theta}} f + \underbrace{(\nabla_{\mathbf{x}} f)^{\top}}_{\mathbf{0} \text{ at optimum}} \frac{\partial \mathbf{x}^*}{\partial \boldsymbol{\theta}}$.

**Application to Bellman equations:** For $[\Bellman v](s) = \max_a \{r(s,a) + \gamma \mathbb{E}[v(s')]\}$, the derivative with respect to parameters in $v$ can be computed by treating the optimal action as constant. For example, if $v(s; \theta) = \sum_j \theta_j \varphi_j(s)$:

$$
\frac{\partial [\Bellman v](s)}{\partial \theta_j} = \gamma \mathbb{E}[\varphi_j(s') \mid s, a^*(s; \theta)],
$$

where $a^*(s; \theta)$ is the optimal action given parameters $\theta$.

**Important assumptions:** The objective $f$ is smooth, the maximizer is unique and in the interior (or constraints are smooth with stable active sets), and the first-order condition holds.
```

With the Envelope Theorem providing a tractable way to compute Jacobians for problems involving optimization, Newton's method becomes practical for weighted residual methods applied to Bellman equations and similar problems. The method offers **quadratic convergence** near the solution. Once in the neighborhood of the true fixed point, Newton's method typically converges in just a few iterations. Unlike function iteration, it doesn't rely on the finite-dimensional approximation preserving any contraction property, making it applicable to a broader range of problems, particularly those with high-order polynomial bases or large discount factors where function iteration struggles.

However, Newton's method demands more from both the algorithm and the user. Each iteration requires computing and solving a full Jacobian system, making the per-iteration cost significantly higher than function iteration. The method is also sensitive to initialization: started far from the solution, Newton's method may diverge or converge to spurious fixed points that the finite-dimensional problem introduces but the original infinite-dimensional problem lacks. When applying the Envelope Theorem, implementation becomes more complex. We must track the optimal action at each evaluation point and compute the Jacobian entries using the formula above (expected basis function values at next states under optimal actions), though the economic interpretation (tracking how value propagates through optimal decisions) often makes the computation conceptually clearer than explicit derivative calculations would be.

##### Comparison and Practical Recommendations

| **Method** | **Convergence** | **Per-iteration cost** | **Initial guess sensitivity** |
|:-----------|:----------------|:-----------------------|:------------------------------|
| **Function iteration** | Linear (when contraction holds) | Low | Robust |
| **Newton's method** | Quadratic (near solution) | Moderate (Jacobian + solve) | Requires good initial guess |

Which method to use? When the problem has strong contraction (small $\gamma$, well-conditioned bases, shape-preserving approximations like linear interpolation or splines), function iteration is simple and robust. For weak contraction (large $\gamma$, high-order polynomials), a hybrid approach works well: run function iteration for several iterations to enter the basin of attraction, then switch to Newton's method for rapid final convergence. When the finite-dimensional approximation destroys contraction entirely (common with non-monotone bases), Newton's method may be necessary from the start, though careful initialization (from a coarser approximation or perturbation methods) is required.

Quasi-Newton methods like BFGS or Broyden offer a middle ground. They approximate the Jacobian using function evaluations only, avoiding explicit derivative computations while maintaining superlinear convergence. This can be useful when computing the exact Jacobian via the Envelope Theorem is expensive or when the approximation quality is acceptable.

### Step 5: Verify the Solution

Once we have computed a candidate solution $\hat{f}$, we must verify its quality. Projection methods optimize $\hat{f}$ with respect to specific criteria (specific test functions or collocation points), but we should check that the residual is small everywhere, including directions or points we did not optimize over.

Typical diagnostic checks include:
- Computing $\|R(\cdot; \theta)\|$ using a more accurate quadrature rule than was used in the optimization
- Evaluating $R(x; \theta)$ at many points not used in the fitting process
- If using Galerkin with the first $n$ basis functions, checking orthogonality against higher-order basis functions

## Application to the Bellman Equation

Consider the Bellman optimality equation $v(s) = \Bellman v(s) = \max_{a \in \mathcal{A}_s} \{ r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) v(j) \}$. For a candidate approximation $\hat{v}(s) = \sum_{i=1}^n \theta_i \varphi_i(s)$, the residual is:

$$
R(s; \theta) = \Bellman\hat{v}(s) - \hat{v}(s) = \max_{a \in \mathcal{A}_s} \left\{ r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) \hat{v}(j) \right\} - \sum_{i=1}^n \theta_i \varphi_i(s).
$$

We examine how collocation and Galerkin, the two most common weighted residual methods for Bellman equations, specialize the general solution approaches from Step 4.

### Collocation

For collocation, we choose $n$ states $\{s_1, \ldots, s_n\}$ and require the Bellman equation to hold exactly at these points:

$$
\sum_{j=1}^n \theta_j \varphi_j(s_i) = \max_{a \in \mathcal{A}_{s_i}} \left\{ r(s_i,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s_i,a) \sum_{\ell=1}^n \theta_\ell \varphi_\ell(j) \right\}, \quad i = 1, \ldots, n.
$$

It helps to define the **parametric Bellman operator** $\mathrm{L}_\varphi: \mathbb{R}^n \to \mathbb{R}^n$ by $[\mathrm{L}_\varphi(\theta)]_i = [\Bellman\hat{v}(\cdot; \theta)](s_i)$, the Bellman operator evaluated at collocation point $s_i$. Let $\boldsymbol{\Phi}$ be the $n \times n$ matrix with entries $\Phi_{ij} = \varphi_j(s_i)$. Then the collocation equations become $\boldsymbol{\Phi} \theta = \mathrm{L}_\varphi(\theta)$.

**Function iteration** for collocation proceeds as follows. Given current coefficients $\theta^{(k)}$, we evaluate the Bellman operator at each collocation point to get target values $t_i^{(k)} = [\mathrm{L}_\varphi(\theta^{(k)})]_i$. We then find new coefficients by solving the linear system $\boldsymbol{\Phi} \theta^{(k+1)} = t^{(k)}$. This is parametric value iteration: apply the Bellman operator, fit the result.

```{prf:algorithm} Collocation with Function Iteration
:label: collocation-function-iteration

**Input** Collocation points $\{s_1, \ldots, s_n\}$, basis functions $\{\varphi_1, \ldots, \varphi_n\}$, initial $\theta^{(0)}$, tolerance $\varepsilon > 0$

**Output** Converged coefficients $\theta^*$

1. Form collocation matrix $\boldsymbol{\Phi}$ with $\Phi_{ij} = \varphi_j(s_i)$
2. $k \leftarrow 0$
3. **repeat**
    1. For each $i = 1, \ldots, n$:
        1. $t_i^{(k)} \leftarrow \max_{a \in \mathcal{A}_{s_i}} \left\{ r(s_i, a) + \gamma \sum_{j \in \mathcal{S}} p(j|s_i, a) \sum_{\ell=1}^n \theta_\ell^{(k)} \varphi_\ell(j) \right\}$
    2. Solve $\boldsymbol{\Phi} \theta^{(k+1)} = t^{(k)}$
    3. $k \leftarrow k + 1$
4. **until** $\|\theta^{(k)} - \theta^{(k-1)}\| < \varepsilon$
5. **return** $\theta^{(k)}$
```

When the state space is continuous, we approximate expectations using numerical quadrature (Gauss-Hermite for normal shocks, etc.). The method is simple and robust when the finite-dimensional approximation preserves contraction, but can be slow for large discount factors.

**Newton's method** for collocation treats the problem as rootfinding: $G(\theta) = \boldsymbol{\Phi} \theta - \mathrm{L}_\varphi(\theta) = 0$. The Jacobian is $J_G = \boldsymbol{\Phi} - J_{\mathrm{L}_\varphi}$, where the Envelope Theorem (Step 4) gives us $[J_{\mathrm{L}_\varphi}]_{ij} = \gamma \sum_{s'} p(s'|s_i, a_i^*(\theta)) \varphi_j(s')$. Here $a_i^*(\theta)$ is the optimal action at collocation point $s_i$ given the current coefficients.

```{prf:algorithm} Collocation with Newton's Method
:label: collocation-newton

**Input** Collocation points $\{s_1, \ldots, s_n\}$, basis functions $\{\varphi_1, \ldots, \varphi_n\}$, initial $\theta^{(0)}$, tolerance $\varepsilon > 0$

**Output** Converged coefficients $\theta^*$

1. Form collocation matrix $\boldsymbol{\Phi}$ with $\Phi_{ij} = \varphi_j(s_i)$
2. $k \leftarrow 0$
3. **repeat**
    1. For each $i = 1, \ldots, n$:
        1. $t_i^{(k)} \leftarrow \max_{a \in \mathcal{A}_{s_i}} \left\{ r(s_i, a) + \gamma \sum_{j \in \mathcal{S}} p(j|s_i, a) \sum_{\ell=1}^n \theta_\ell^{(k)} \varphi_\ell(j) \right\}$
        2. Store $a_i^* \in \arg\max$ achieving the maximum
    2. Compute Jacobian: $[J_{\mathrm{L}_\varphi}]_{ij} = \gamma \sum_{j \in \mathcal{S}} p(j|s_i, a_i^*) \varphi_j(j)$ for all $i,j$
    3. Solve $(\boldsymbol{\Phi} - J_{\mathrm{L}_\varphi}) \Delta\theta = \boldsymbol{\Phi} \theta^{(k)} - t^{(k)}$
    4. $\theta^{(k+1)} \leftarrow \theta^{(k)} - \Delta\theta$
    5. $k \leftarrow k + 1$
4. **until** $\|\Delta\theta\| < \varepsilon$
5. **return** $\theta^{(k)}$
```

This converges rapidly near the solution but requires good initialization and more computation per iteration than function iteration. The method is equivalent to policy iteration: each step evaluates the value of the current greedy policy, then improves it.

Why is collocation popular for Bellman equations? Because it avoids integration when testing the residual. We only evaluate the Bellman operator at $n$ discrete points. In contrast, Galerkin requires integrating the residual against each basis function.

### Galerkin

For Galerkin, we use the basis functions themselves as test functions. The conditions are:

$$
\int_{\mathcal{S}} [\Bellman\hat{v}(s; \theta) - \hat{v}(s; \theta)] \varphi_i(s) w(s) ds = 0, \quad i = 1, \ldots, n.
$$

where $w(s)$ is a weight function (often the stationary distribution $d^\pi(s)$ in RL applications, or simply $w(s) = 1$). Expanding this:

$$
\int_{\mathcal{S}} \left[ \max_a \left\{ r(s,a) + \gamma \mathbb{E}[v(s')] \right\} - \sum_j \theta_j \varphi_j(s) \right] \varphi_i(s) w(s) ds = 0.
$$

**Function iteration** for Galerkin works differently than for collocation. Given $\theta^{(k)}$, we cannot simply evaluate the Bellman operator and fit. Instead, we must solve an integral equation. At each iteration, we seek $\theta^{(k+1)}$ satisfying:

$$
\int_{\mathcal{S}} \sum_j \theta_j^{(k+1)} \varphi_j(s) \varphi_i(s) w(s) ds = \int_{\mathcal{S}} [\Bellman\hat{v}(s; \theta^{(k)})] \varphi_i(s) w(s) ds.
$$

```{prf:algorithm} Galerkin with Function Iteration
:label: galerkin-function-iteration

**Input** Basis functions $\{\varphi_1, \ldots, \varphi_n\}$, weight function $w(s)$, initial $\theta^{(0)}$, tolerance $\varepsilon > 0$

**Output** Converged coefficients $\theta^*$

1. Compute mass matrix $M_{ij} = \int_{\mathcal{S}} \varphi_i(s) \varphi_j(s) w(s) ds$ via numerical integration
2. $k \leftarrow 0$
3. **repeat**
    1. For each $i = 1, \ldots, n$:
        1. $b_i^{(k)} \leftarrow \int_{\mathcal{S}} [\Bellman\hat{v}(s; \theta^{(k)})] \varphi_i(s) w(s) ds$ via numerical integration
    2. Solve $M \theta^{(k+1)} = b^{(k)}$
    3. $k \leftarrow k + 1$
4. **until** $\|\theta^{(k)} - \theta^{(k-1)}\| < \varepsilon$
5. **return** $\theta^{(k)}$
```

The left side is a linear system (the "mass matrix" $M_{ij} = \int \varphi_i \varphi_j w$), and the right side requires integrating the Bellman operator output against each test function. When the basis functions are orthogonal polynomials with matching weight $w$, the mass matrix is diagonal, simplifying the solve. But we still need numerical integration to evaluate the right side. This makes Galerkin substantially more expensive than collocation per iteration.

**Newton's method** for Galerkin similarly requires integration. The residual is $R(s; \theta) = \Bellman\hat{v}(s; \theta) - \hat{v}(s; \theta)$, and we need $G_i(\theta) = \int R(s; \theta) \varphi_i(s) w(s) ds = 0$. The Jacobian entry is:

$$
J_{ij} = \int \left[ \frac{\partial \Bellman\hat{v}(s; \theta)}{\partial \theta_j} - \varphi_j(s) \right] \varphi_i(s) w(s) ds.
$$

```{prf:algorithm} Galerkin with Newton's Method
:label: galerkin-newton

**Input** Basis functions $\{\varphi_1, \ldots, \varphi_n\}$, weight function $w(s)$, initial $\theta^{(0)}$, tolerance $\varepsilon > 0$

**Output** Converged coefficients $\theta^*$

1. $k \leftarrow 0$
2. **repeat**
    1. For each $i = 1, \ldots, n$:
        1. $G_i^{(k)} \leftarrow \int_{\mathcal{S}} [\Bellman\hat{v}(s; \theta^{(k)}) - \hat{v}(s; \theta^{(k)})] \varphi_i(s) w(s) ds$
        2. For each $j = 1, \ldots, n$:
            1. $J_{ij} \leftarrow \int_{\mathcal{S}} \left[ \gamma \mathbb{E}[\varphi_j(s') \mid s, a^*(s;\theta^{(k)})] - \varphi_j(s) \right] \varphi_i(s) w(s) ds$
    2. Solve $J \Delta\theta = G^{(k)}$
    3. $\theta^{(k+1)} \leftarrow \theta^{(k)} - \Delta\theta$
    4. $k \leftarrow k + 1$
3. **until** $\|\Delta\theta\| < \varepsilon$
4. **return** $\theta^{(k)}$
```

The Envelope Theorem gives $\frac{\partial \Bellman\hat{v}(s; \theta)}{\partial \theta_j} = \gamma \mathbb{E}[\varphi_j(s') \mid s, a^*(s;\theta)]$, so we must integrate expected basis function values (under optimal actions) against test functions and weight. This requires both numerical integration and careful tracking of optimal actions across the state space, making it substantially more complex than collocation's pointwise evaluation.

The advantage of Galerkin over collocation lies in its theoretical properties: when using orthogonal polynomials, Galerkin provides optimal approximation in the weighted $L^2$ norm. For smooth problems, this can yield better accuracy per degree of freedom than collocation. In practice, collocation's computational simplicity usually outweighs Galerkin's theoretical optimality for Bellman equations, especially in high-dimensional problems where integration becomes prohibitively expensive.

### Galerkin for Discrete MDPs: LSTD and LSPI

When the state space is discrete and finite, the Galerkin conditions simplify dramatically. The integrals become sums, and we can write everything in matrix form. This specialization shows the connection to algorithms widely used in reinforcement learning.

For a discrete state space $\mathcal{S} = \{s_1, \ldots, s_m\}$, the Galerkin orthogonality conditions

$$
\int_{\mathcal{S}} [\Bellman\hat{v}(s; \theta) - \hat{v}(s; \theta)] \varphi_i(s) w(s) ds = 0
$$

become weighted sums over states:

$$
\sum_{s \in \mathcal{S}} \xi(s) [\Bellman\hat{v}(s; \theta) - \hat{v}(s; \theta)] \varphi_i(s) = 0, \quad i = 1, \ldots, n,
$$

where $\xi(s) \geq 0$ with $\sum_s \xi(s) = 1$ is a probability distribution over states. Define the feature matrix $\boldsymbol{\Phi} \in \mathbb{R}^{m \times n}$ with entries $\Phi_{si} = \varphi_i(s)$ (each row contains the features for one state), and let $\boldsymbol{\Xi} = \text{diag}(\xi)$ be the diagonal matrix with the state distribution on the diagonal.

#### Policy Evaluation: LSTD

For **policy evaluation** with a fixed policy $\pi$, the Bellman operator is linear:

$$
[\BellmanPi \hat{v}](s) = r(s, \pi(s)) + \gamma \sum_{j \in \mathcal{S}} p(j|s, \pi(s)) \hat{v}(j).
$$

With linear function approximation $\hat{v}(s) = \boldsymbol{\varphi}(s)^\top \theta = \sum_i \theta_i \varphi_i(s)$, this becomes:

$$
[\BellmanPi \hat{v}](s) = r(s, \pi(s)) + \gamma \sum_{j \in \mathcal{S}} p(j|s, \pi(s)) \sum_i \theta_i \varphi_i(j).
$$

Let $\mathbf{r}_\pi \in \mathbb{R}^m$ be the vector of rewards $[\mathbf{r}_\pi]_s = r(s, \pi(s))$, and $\mathbf{P}_\pi \in \mathbb{R}^{m \times m}$ be the transition matrix with $[\mathbf{P}_\pi]_{sj} = p(j|s, \pi(s))$. Then $\BellmanPi \hat{v} = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi \boldsymbol{\Phi} \theta$ in vector form.

The Galerkin conditions require $\langle \BellmanPi \hat{v} - \hat{v}, \varphi_i \rangle_\xi = 0$ for all basis functions, which in matrix form is:

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\mathbf{r}_\pi + \gamma \mathbf{P}_\pi \boldsymbol{\Phi} \theta - \boldsymbol{\Phi} \theta) = \mathbf{0}.
$$

Rearranging:

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_\pi \boldsymbol{\Phi}) \theta = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{r}_\pi.
$$ (eq:lstd-galerkin)

This is the **LSTD (Least Squares Temporal Difference)** solution. The matrix $\mathbf{A} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_\pi \boldsymbol{\Phi})$ and vector $\mathbf{b} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{r}_\pi$ give the linear system $\mathbf{A} \theta = \mathbf{b}$.

When $\xi$ is the stationary distribution of policy $\pi$ (so $\xi^\top \mathbf{P}_\pi = \xi^\top$), this system has a unique solution, and the projected Bellman operator $\Proj \BellmanPi$ is a contraction in the weighted $L^2$ norm $\|\cdot\|_\xi$. This is the theoretical foundation for TD learning with linear function approximation.

#### The Bellman Optimality Equation: Function Iteration and Newton's Method

For the **Bellman optimality equation**, the max operator introduces nonlinearity:

$$
[\Bellman\hat{v}](s) = \max_{a \in \mathcal{A}_s} \left\{ r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) \hat{v}(j) \right\}.
$$

The Galerkin conditions become:

$$
F(\theta) \equiv \boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\Bellman\hat{v}(\cdot; \theta) - \boldsymbol{\Phi} \theta) = \mathbf{0},
$$

where the Bellman operator must be evaluated at each state $s$ to find the optimal action and compute the target value. This is a system of $n$ nonlinear equations in $n$ unknowns.

**Function iteration** applies the Bellman operator and projects back. Given $\theta^{(k)}$, compute the greedy policy $\pi^{(k)}(s) = \arg\max_a \{r(s,a) + \gamma \sum_{j \in \mathcal{S}} p(j|s,a) \boldsymbol{\varphi}(j)^\top \theta^{(k)}\}$ at each state, then solve:

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_{\pi^{(k)}} \boldsymbol{\Phi}) \theta^{(k+1)} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{r}_{\pi^{(k)}}.
$$

This evaluates the current greedy policy using LSTD, then implicitly improves by computing a new greedy policy at the next iteration. However, convergence can be slow when the finite-dimensional approximation poorly preserves contraction.

**Newton's method** treats $G(\theta) = 0$ as a rootfinding problem and uses the Jacobian to accelerate convergence. The Jacobian of $G$ is:

$$
J_G(\theta) = \frac{\partial G}{\partial \theta} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \left( \frac{\partial \Bellman\hat{v}(\cdot; \theta)}{\partial \theta} - \boldsymbol{\Phi} \right).
$$

To compute $\frac{\partial \Bellman\hat{v}(s; \theta)}{\partial \theta_j}$, we use the Envelope Theorem from Step 4. At the current $\theta^{(k)}$, let $a^*(s; \theta^{(k)})$ be the optimal action at state $s$. Then:

$$
\frac{\partial [\Bellman\hat{v}](s; \theta^{(k)})}{\partial \theta_j} = \gamma \sum_{j \in \mathcal{S}} p(j|s, a^*(s; \theta^{(k)})) \varphi_j(j).
$$

Define the policy $\pi^{(k)}(s) = a^*(s; \theta^{(k)})$. The Jacobian becomes:

$$
J_G(\theta^{(k)}) = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\gamma \mathbf{P}_{\pi^{(k)}} \boldsymbol{\Phi} - \boldsymbol{\Phi}) = -\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_{\pi^{(k)}} \boldsymbol{\Phi}).
$$

The Newton update $\theta^{(k+1)} = \theta^{(k)} - J_G(\theta^{(k)})^{-1} G(\theta^{(k)})$ simplifies. We have:

$$
G(\theta^{(k)}) = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\Bellman\hat{v}(\cdot; \theta^{(k)}) - \boldsymbol{\Phi} \theta^{(k)}).
$$

At each state $s$, the greedy value is $[\Bellman\hat{v}(\cdot; \theta^{(k)})](s) = r(s, \pi^{(k)}(s)) + \gamma \sum_j p(j|s, \pi^{(k)}(s)) \boldsymbol{\varphi}(j)^\top \theta^{(k)}$, which equals $[\mathrm{L}_{\pi^{(k)}} \hat{v}(\cdot; \theta^{(k)})](s)$. Thus:

$$
G(\theta^{(k)}) = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\mathbf{r}_{\pi^{(k)}} + \gamma \mathbf{P}_{\pi^{(k)}} \boldsymbol{\Phi} \theta^{(k)} - \boldsymbol{\Phi} \theta^{(k)}).
$$

The Newton step becomes:

$$
\theta^{(k+1)} = \theta^{(k)} + [\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_{\pi^{(k)}} \boldsymbol{\Phi})]^{-1} \boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\mathbf{r}_{\pi^{(k)}} + \gamma \mathbf{P}_{\pi^{(k)}} \boldsymbol{\Phi} \theta^{(k)} - \boldsymbol{\Phi} \theta^{(k)}).
$$

Multiplying through and simplifying:

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_{\pi^{(k)}} \boldsymbol{\Phi}) \theta^{(k+1)} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{r}_{\pi^{(k)}}.
$$

This is **LSPI (Least Squares Policy Iteration)**. Each Newton step:
1. Computes the greedy policy $\pi^{(k)}(s) = \arg\max_a \{r(s,a) + \gamma \sum_j p(j|s,a) \boldsymbol{\varphi}(j)^\top \theta^{(k)}\}$
2. Solves the LSTD equation for this policy to get $\theta^{(k+1)}$

Newton's method for the Galerkin-projected Bellman optimality equation is equivalent to policy iteration in the function approximation setting. Just as Newton's method for collocation corresponded to policy iteration (Step 4), Newton's method for discrete Galerkin gives LSPI.

Galerkin projection with linear function approximation reduces policy iteration to a sequence of linear systems, each solvable in closed form. For discrete MDPs, we can compute the matrices $\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \boldsymbol{\Phi}$ and $\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{P}_\pi \boldsymbol{\Phi}$ exactly.

## Fitted-Value/Q Iteration (FVI/FQI)

We have developed weighted residual methods through abstract functional equations: choose test functions, impose orthogonality conditions $\langle R, p_i \rangle_w = 0$, solve for coefficients. But what are we actually computing when we solve these equations by successive approximation? The answer is simpler than the formalism suggests: **function iteration with a fitting step**.

Recall that the weighted residual conditions $\langle v - \Bellman v, p_i \rangle_w = 0$ define a fixed-point problem $v = \Proj \Bellman v$, where $\Proj$ is a projection operator onto $\text{span}(\boldsymbol{\Phi})$. We can solve this by iteration: $v_{k+1} = \Proj \Bellman v_k$. Under appropriate conditions (monotonicity of $\Proj$, or matching the weight to the operator for policy evaluation), this converges to a solution.

In parameter space, this iteration becomes a fitting procedure. Consider Galerkin projection with a finite state space of $n$ states. Let $\boldsymbol{\Phi}$ be the $n \times d$ matrix of basis evaluations, $\mathbf{W}$ the diagonal weight matrix, and $\mathbf{y}$ the vector of Bellman operator evaluations: $y_i = (\Bellman v_k)(s_i)$. The projection is:

$$
\boldsymbol{\theta}_{k+1} = (\boldsymbol{\Phi}^\top \mathbf{W} \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \mathbf{W} \mathbf{y}.
$$

This is weighted least-squares regression: fit $\boldsymbol{\Phi} \boldsymbol{\theta}$ to targets $\mathbf{y}$. For collocation, we require exact interpolation $\boldsymbol{\Phi} \boldsymbol{\theta}_{k+1} = \mathbf{y}$ at chosen collocation points. For continuous state spaces, we approximate the Galerkin integrals using sampled states, reducing to the same finite-dimensional fitting problem. The abstraction remains consistent: function iteration in the abstract becomes **generate targets, fit to targets, repeat** in the implementation.

This extends beyond linear basis functions. Neural networks, decision trees, and kernel methods all implement variants of this procedure. Given data $\{(s_i, y_i)\}$ where $y_i = (\Bellman v_k)(s_i)$, each method produces a function $v_{k+1}: \mathcal{S} \to \mathbb{R}$ by fitting to the targets. The projection operator $\Proj$ is simply one instantiation of a fitting procedure. Galerkin and collocation correspond to specific choices of approximation class and loss function.

```{prf:algorithm} Fitted-Value Iteration
:label: fitted-value-iteration

**Inputs:** Finite state set $\mathcal{S}$ (or sample $\{s_i\}_{i=1}^n$), discount factor $\gamma$, function class $\mathcal{F}$, fitting procedure $\mathtt{fit}$, convergence tolerance $\epsilon$

**Output:** Approximate value function $\hat{v} \approx v^*$

1. Initialize $v_0 \in \mathcal{F}$ arbitrarily
2. Set $k \leftarrow 0$
3. **repeat**
4. $\quad$ **for** each state $s_i \in \mathcal{S}$ **do**
5. $\quad\quad$ Compute target: $y_i \leftarrow \displaystyle\max_{a \in \mathcal{A}} \Big\{ r(s_i, a) + \gamma \sum_{s'} p(s' \mid s_i, a) v_k(s') \Big\}$
6. $\quad$ **end for**
7. $\quad$ Fit new approximation: $v_{k+1} \leftarrow \mathtt{fit}\big(\{(s_i, y_i)\}_{i=1}^n; \mathcal{F}\big)$
8. $\quad$ $k \leftarrow k+1$
9. **until** $\|v_k - v_{k-1}\| < \epsilon$ (or maximum iterations reached)
10. **return** $v_k$
```

The abstraction $\mathtt{fit}$ encapsulates all the complexity of function approximation, whether that involves solving a linear system, running gradient descent, or training an ensemble. The projection operator $\Proj$ is one instantiation: when $\mathcal{F}$ is a linear subspace and we minimize weighted squared error, we recover Galerkin or collocation. Neural networks and other non-linear methods extend this framework beyond theoretical tractability.

### Extension to Nonlinear Approximators

The weighted residual methods developed so far have focused on linear function classes: polynomial bases, piecewise linear interpolants, and linear combinations of fixed basis functions. Neural networks, kernel methods, and decision trees do not fit this template. How does the framework extend to nonlinear approximators?

Recall the Galerkin approach for linear approximation $v_{\boldsymbol{\theta}} = \sum_{i=1}^d \theta_i \varphi_i$. The orthogonality conditions $\langle v - \Bellman v, \varphi_i \rangle_w = 0$ for all $i$ define a linear system with a closed-form solution. These equations arise from minimizing $\|v - \Bellman v\|_w^2$ over the subspace, since at the minimum, the gradient with respect to each coefficient must vanish. The connection between norm minimization and orthogonality holds generally. For any norm $\|\cdot\|_w$ induced by an inner product $\langle \cdot, \cdot \rangle_w$, minimizing $\|f(\boldsymbol{\theta})\|_w^2$ with respect to parameters requires $\frac{\partial}{\partial \theta_i} \|f(\boldsymbol{\theta})\|_w^2 = 0$. Since $\|f\|_w^2 = \langle f, f \rangle_w$, the chain rule gives $2\langle f, \frac{\partial f}{\partial \theta_i} \rangle_w = 0$. Minimizing the residual norm is thus equivalent to requiring orthogonality $\langle f, \frac{\partial f}{\partial \theta_i} \rangle_w = 0$ for all $i$. The equivalence holds for any choice of inner product: weighted $L^2$ integrals for Galerkin, sums over collocation points for collocation, or sampled expectations for neural networks.

For nonlinear function classes parameterized by $\boldsymbol{\theta} \in \mathbb{R}^p$ (neural networks, kernel expansions), the same minimization principle applies:

$$
\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \|v_{\boldsymbol{\theta}} - \Bellman v_{\boldsymbol{\theta}}\|_w^2.
$$

The first-order stationarity condition yields orthogonality:

$$
\Big\langle v_{\boldsymbol{\theta}} - \Bellman v_{\boldsymbol{\theta}}, \frac{\partial v_{\boldsymbol{\theta}}}{\partial \theta_i} \Big\rangle_w = 0 \quad \text{for all } i.
$$

The test functions are now the partial derivatives $\frac{\partial v_{\boldsymbol{\theta}}}{\partial \theta_i}$, which span the tangent space to the manifold $\{v_{\boldsymbol{\theta}} : \boldsymbol{\theta} \in \mathbb{R}^p\}$ at the current parameters. In the linear case $v_{\boldsymbol{\theta}} = \sum_i \theta_i \varphi_i$, the partial derivative $\frac{\partial v_{\boldsymbol{\theta}}}{\partial \theta_i} = \varphi_i$ recovers the fixed basis functions of Galerkin. For nonlinear parameterizations, the test functions change with $\boldsymbol{\theta}$, and the orthogonality conditions define a nonlinear system solved by iterative gradient descent.

The **dual pairing** formulation {cite}`LegrandJunca2025` extends this framework to settings where test objects need not be regular functions. We have been informal about this distinction in our treatment of collocation, but the Dirac deltas $\delta(x - x_i)$ used there are not classical functions. They are distributions, defined rigorously only through their action on test functions via $\langle \Residual(v), \delta(x - x_i) \rangle = (\Residual v)(x_i)$. The simple calculus argument for orthogonality does not apply directly to such objects; the dual pairing framework provides the proper mathematical foundation. The induced dual norm $\|\Residual(v)\|_* = \sup_{\|w\|=1} |\langle \Residual(v), w \rangle|$ measures residuals by their worst-case effect on test functions, a perspective that has inspired adversarial formulations {cite}`Zang2020` where both trial and test functions are learned.

The minimum residual framework thus connects classical projection methods to modern function approximation. The unifying principle is orthogonality of residuals to test functions. Linear methods use fixed test functions and admit closed-form solutions. Nonlinear methods use parameter-dependent test functions and require iterative optimization.

A limitation of FVI/FQI is that it assumes we can evaluate the Bellman operator exactly. Computing $y_i = (\Bellman v_k)(s_i)$ requires knowing transition probabilities and summing over all next states. In practice, we often have only a simulator or observed data. A later chapter shows how to approximate these expectations from samples, connecting the fitted-value iteration framework to simulation-based methods.

We now turn to the question of convergence: when does the iteration $v_{k+1} = \Proj \Bellman v_k$ converge?

## Monotone Projection and the Preservation of Contraction

The informal discussion of shape preservation hints at a deeper theoretical question: **when does the function iteration method converge?** Recall from our discussion of collocation that function iteration proceeds in two steps:

1. Apply the Bellman operator at collocation points: $t^{(k)} = v(\theta^{(k)})$ where $t_i^{(k)} = \Bellman\hat{v}^{(k)}(s_i)$
2. Fit new coefficients to match these targets: $\boldsymbol{\Phi} \theta^{(k+1)} = t^{(k)}$, giving $\theta^{(k+1)} = \boldsymbol{\Phi}^{-1} v(\theta^{(k)})$

We can reinterpret this iteration in **function space** rather than coefficient space. Let $\Proj$ be the **projection operator** that takes any function $f$ and returns its approximation in $\text{span}\{\varphi_1, \ldots, \varphi_n\}$. For collocation, $\Proj$ is the interpolation operator: $(\Proj f)(s)$ is the unique linear combination of basis functions that matches $f$ at the collocation points. Then Step 2 can be written as: fit $\hat{v}^{(k+1)}$ so that $\hat{v}^{(k+1)}(s_i) = \Bellman\hat{v}^{(k)}(s_i)$ for all collocation points, which means $\hat{v}^{(k+1)} = \Proj(\Bellman\hat{v}^{(k)})$.

In other words, function iteration is equivalent to **projected value iteration in function space**:

$$
\hat{v}^{(k+1)} = \Proj \Bellman \hat{v}^{(k)}.
$$

We know that standard value iteration $v_{k+1} = \Bellman v_k$ converges because $\Bellman$ is a $\gamma$-contraction in the sup norm. But now we're iterating with the **composed operator** $\Proj \Bellman$ instead of $\Bellman$ alone.

This $\Proj \Bellman$ structure is not specific to collocation. It is inherent in all projection methods. The general pattern is always the same: apply the Bellman operator to get a target function $\Bellman\hat{v}^{(k)}$, then project it back onto our approximation space to get $\hat{v}^{(k+1)}$. The projection step defines an operator $\Proj$ that depends on our choice of test functions:

- For **collocation**, $\Proj$ interpolates values at collocation points
- For **Galerkin**, $\Proj$ is orthogonal projection with respect to $\langle \cdot, \cdot \rangle_w$  
- For **least squares**, $\Proj$ minimizes the weighted residual norm

But regardless of which projection method we use, iteration takes the form $\hat{v}^{(k+1)} = \Proj \Bellman\hat{v}^{(k)}$.

**Does the composition $\Proj \Bellman$ inherit the contraction property of $\Bellman$?** If not, the iteration may diverge, oscillate, or converge to a spurious fixed point even though the original problem is well-posed.

### Monotone Approximators and Stability

The answer turns out to depend on specific properties of the approximation operator $\Proj$. This theory was developed independently across multiple research communities: computational economics {cite}`Judd1992,Judd1996,McGrattan1997,SantosVigoAguiar1998`, economic dynamics {cite}`Stachurski2009`, and reinforcement learning {cite}`Gordon1995,Gordon1999`. These communities arrived at essentially the same mathematical conditions.

#### Monotonicity Implies Nonexpansiveness

It turns out that approximation operators satisfying simple structural properties automatically preserve contraction.

```{prf:proposition} Monotone operators are nonexpansive (Stachurski)
:label: monotone-nonexpansive

Let $\Proj: \mathcal{V} \to \mathcal{V}$ be a linear operator on the space $\mathcal{V}$ of bounded real-valued functions on $\mathcal{S}$. If $\Proj$ satisfies:

1. **Monotonicity**: $f \leq g$ pointwise implies $\Proj f \leq \Proj g$
2. **Constant preservation**: $\Proj\mathbf{1} = \mathbf{1}$ where $\mathbf{1}$ is the constant function equal to $1$

Then $\Proj$ is nonexpansive in the sup norm: $\|\Proj f - \Proj g\|_\infty \leq \|f - g\|_\infty$ for all $f, g \in \mathcal{V}$.
```

```{prf:proof}
Let $M = \|f - g\|_\infty$. Then $-M \leq f(s) - g(s) \leq M$ for all $s$, which can be written as $g - M\mathbf{1} \leq f \leq g + M\mathbf{1}$. By monotonicity, $\Proj(g - M\mathbf{1}) \leq \Proj f \leq \Proj(g + M\mathbf{1})$. By linearity and constant preservation, $\Proj g - M\mathbf{1} \leq \Proj f \leq \Proj g + M\mathbf{1}$, which means $|\Proj f(s) - \Proj g(s)| \leq M$ for all $s$. Therefore $\|\Proj f - \Proj g\|_\infty \leq \|f - g\|_\infty$.
```

This proposition shows that monotonicity and constant preservation automatically imply nonexpansiveness. There is no need to verify this separately. The intuition is that a monotone, constant-preserving operator acts like a weighted average that respects order structure and cannot amplify differences between functions.

#### Preservation of Contraction

Combining nonexpansiveness with the contraction property of the Bellman operator yields the main stability result.

```{prf:theorem} Stability of projected value iteration (Santos-Vigo-Aguiar)
:label: santos-vigo-aguiar-stability

Let $\Bellman: \mathcal{V} \to \mathcal{V}$ be a $\gamma$-contraction on the space $\mathcal{V}$ of bounded functions with respect to the sup norm. Let $\Proj: \mathcal{V} \to \mathcal{V}$ be a linear approximation operator satisfying monotonicity and constant preservation.

Then the composed operator $\Proj \Bellman$ is a $\gamma$-contraction, and projected value iteration $v_{k+1} = \Proj \Bellman v_k$ converges globally to a unique fixed point $v_\Proj \in \text{Range}(\Proj)$ with approximation error:

$$
\|v_\Proj - v^*\|_\infty \leq \frac{1}{1-\gamma} \|\Proj v^* - v^*\|_\infty,
$$

where $v^*$ is the true value function.
```

```{prf:proof}
By {prf:ref}`monotone-nonexpansive`, $\Proj$ is nonexpansive since it satisfies monotonicity and constant preservation. Since $\Bellman$ is a $\gamma$-contraction, we have $\|\Bellman f - \Bellman g\|_\infty \leq \gamma\|f-g\|_\infty$. Therefore:

$$
\|\Proj \Bellman f - \Proj \Bellman g\|_\infty \leq \|\Bellman f - \Bellman g\|_\infty \leq \gamma\|f-g\|_\infty,
$$

showing that $\Proj \Bellman$ is a $\gamma$-contraction. The error bound follows from fixed-point analysis: $v^* - v_\Proj = (I - \Proj \Bellman)^{-1}(v^* - \Proj v^*)$, and since $\Proj \Bellman$ is a $\gamma$-contraction, $\|(I - \Proj \Bellman)^{-1}\| \leq (1-\gamma)^{-1}$.
```

This error bound tells us that the fixed-point error is controlled by how well $\Proj$ can represent $v^*$. If $v^* \in \text{Range}(\Proj)$, then $\Proj v^* = v^*$ and the error vanishes. Otherwise, the error is proportional to the approximation error $\|\Proj v^* - v^*\|_\infty$, amplified by the factor $(1-\gamma)^{-1}$.

#### Averagers in Discrete-State Problems

For discrete-state problems, the monotonicity conditions have a natural interpretation as **averaging with nonnegative weights**. This characterization was developed by Gordon in the context of reinforcement learning.

```{prf:definition} Averager (Gordon)
:label: gordon-averager

An operator $\Proj: \mathbb{R}^{|\mathcal{S}|} \to \mathbb{R}^{|\mathcal{S}|}$ is an **averager** if $\Proj v = Wv$ where $W$ is a $|\mathcal{S}| \times |\mathcal{S}|$ stochastic matrix: $w_{ij} \geq 0$ and $\sum_j w_{ij} = 1$ for all $i$.
```

Averagers automatically satisfy the monotonicity conditions: linearity follows from matrix multiplication, monotonicity follows from nonnegativity of entries, and constant preservation follows from row sums equaling one.

```{prf:theorem} Stability with averagers (Gordon)
:label: gordon-stability

If $\Proj$ is an averager and $\Bellman$ is the Bellman operator (a $\gamma$-contraction), then $\Proj \Bellman$ is a $\gamma$-contraction, and value iteration $v_{k+1} = \Proj \Bellman v_k$ converges to a unique fixed point.
```

This specializes the Santos-Vigo-Aguiar theorem to discrete states, expressed in the probabilistic language of stochastic matrices. The stochastic matrix characterization connects to Markov chain theory: $\Proj v$ represents expected values after one transition, and the monotonicity property reflects the fact that expectations preserve order.

**Examples of averagers** include state aggregation (averaging values within groups), K-nearest neighbors (averaging over nearest states), kernel smoothing with positive kernels, and multilinear interpolation on grids (barycentric weights are nonnegative and sum to one). **Counterexamples** include linear least squares regression (projection matrix may have negative entries) and high-order polynomial interpolation (Runge phenomenon produces negative weights).

The following table summarizes which common approximation operators satisfy the monotonicity conditions:

| **Method** | **Monotone?** | **Notes** |
|:-----------|:--------------|:----------|
| Piecewise linear interpolation | Yes | Always an averager; guaranteed stability |
| Multilinear interpolation (grid) | Yes | Barycentric weights are nonnegative and sum to one |
| Shape-preserving splines (Schumaker) | Yes | Designed to maintain monotonicity |
| State aggregation | Yes | Exact averaging within groups |
| Kernel smoothing (positive kernels) | Yes | If kernel integrates to one |
| High-order polynomial interpolation | No | Oscillations violate monotonicity (Runge phenomenon) |
| Least squares projection (arbitrary basis) | No | Projection matrix may have negative entries |
| Fourier/spectral methods | No | Not monotone-preserving in general |
| Neural networks | No | Highly flexible but no monotonicity guarantees |

The distinction between "safe" (monotone) and "potentially unstable" (non-monotone) approximators provides rigorous foundation for the folk wisdom that linear interpolation is reliable while high-order polynomials can be dangerous for value iteration. But notice that the table's verdict on "least squares projection" is somewhat abstract. It doesn't specifically address the three weighted residual methods we introduced at the start of this chapter.

The choice of solution method determines which approximation operators are safe to use. Successive approximation (fixed-point iteration) requires monotone approximators to guarantee convergence. Rootfinding methods like Newton's method do not require monotonicity. Stability depends on numerical properties of the Jacobian rather than contraction preservation. These considerations suggest hybrid strategies. One approach runs a few iterations with a monotone method to generate an initial guess, then switches to Newton's method with a smooth approximation for rapid final convergence. 

### Connecting Back to Collocation, Galerkin, and Least Squares

We have now developed a general stability theory for projected value iteration and surveyed which approximation operators are monotone. But what does this mean for the three specific weighted residual methods we introduced at the start of this chapter: **collocation**, **Galerkin**, and **least squares**? Each method defines a different projection operator $\Proj$, and we now need to determine which satisfy the monotonicity conditions that guarantee convergence.

Collocation with piecewise linear interpolation is monotone. When we use collocation with piecewise linear basis functions on a grid, the projection operator performs linear interpolation between grid points. At any state $s$ between grid points $s_i$ and $s_{i+1}$, the interpolated value is:

$$
(\Proj v)(s) = \frac{s_{i+1} - s}{s_{i+1} - s_i} v(s_i) + \frac{s - s_i}{s_{i+1} - s_i} v(s_{i+1}).
$$

The interpolation weights (barycentric coordinates) are nonnegative and sum to one, making this an averager in Gordon's sense. Therefore collocation with piecewise linear bases satisfies the monotonicity conditions and the Santos-Vigo-Aguiar stability theorem applies. The folk wisdom that "linear interpolation is safe for value iteration" has rigorous theoretical foundation.

Galerkin projection is generally not monotone. The Galerkin projection operator for a general basis $\{\varphi_1, \ldots, \varphi_n\}$ has the form:

$$
\Proj = \boldsymbol{\Phi}(\boldsymbol{\Phi}^\top \mathbf{W} \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \mathbf{W},
$$

where $\mathbf{W}$ is a diagonal weight matrix and $\boldsymbol{\Phi}$ contains the basis function evaluations. This projection matrix typically has **negative entries**. To see why, consider a simple example with polynomial basis functions $\{1, x, x^2\}$ on $[-1, 1]$. The projection of a function onto this space involves computing $(\boldsymbol{\Phi}^\top \mathbf{W} \boldsymbol{\Phi})^{-1}$, and the resulting operator can map nonnegative functions to functions with negative values. This is the same phenomenon underlying the Runge phenomenon in high-order polynomial interpolation: the projection weights oscillate in sign.

Since Galerkin projection is not monotone, the sup norm contraction theory does not guarantee convergence of projected value iteration $v_{k+1} = \Proj \Bellman v_k$ with Galerkin.

Least squares methods share the non-monotonicity issue. The least squares projection operator minimizes $\|\Residual(\hat{f})\|_w^2$ and has the same mathematical form as Galerkin projection. It is a linear projection onto $\text{span}\{\varphi_1, \ldots, \varphi_n\}$ with respect to a weighted inner product. Like Galerkin, the projection matrix typically contains negative entries and violates monotonicity.

The monotone approximator framework successfully covers collocation with simple bases, but leaves two important methods, Galerkin and least squares, without convergence guarantees. These methods are used in least-squares temporal difference learning (LSTD) and modern reinforcement learning with linear function approximation. We need a different analytical framework to understand when these non-monotone projections lead to convergent algorithms.

## Beyond Monotone Approximators

The monotone approximator theory gives us a clean sufficient condition for convergence: if $\Proj$ is monotone (and constant-preserving), then $\Proj$ is non-expansive in the sup norm $\|\cdot\|_\infty$. Since $\Bellman$ is a $\gamma$-contraction in the sup norm, their composition $\Proj \Bellman$ is also a $\gamma$-contraction in the sup norm, guaranteeing convergence of projected value iteration.

But what if $\Proj$ is not monotone? Can we still guarantee convergence? Galerkin and least squares projections typically violate monotonicity, yet they are widely used in practice, particularly in reinforcement learning through least-squares temporal difference learning (LSTD). In general, proving convergence for non-monotone projections is difficult. However, for the special case of **policy evaluation**, computing the value function $v_\pi$ of a fixed policy $\pi$, we can establish convergence by working in a different norm.

### The Policy Evaluation Problem and LSTD

Consider the policy evaluation problem: given policy $\pi$, we want to solve the policy Bellman equation $v_\pi = r_\pi + \gamma \mathbf{P}_\pi v_\pi$, where $r_\pi$ and $\mathbf{P}_\pi$ are the reward vector and transition matrix under $\pi$. This is the core computational task in policy iteration, actor-critic algorithms, and temporal difference learning. In reinforcement learning, we typically learn from sampled experience: trajectories $(s_0, a_0, r_1, s_1, a_1, r_2, s_2, \ldots)$ generated by following $\pi$. If the Markov chain induced by $\pi$ is ergodic, the state distribution converges to a stationary distribution $\xi$ satisfying $\xi^\top \mathbf{P}_\pi = \xi^\top$.

This distribution determines which states appear frequently in our data. States visited often contribute more samples and have more influence on any learned approximation. States visited rarely contribute little. For a linear approximation $v_\theta(s) = \sum_j \theta_j \varphi_j(s)$, the **least-squares temporal difference (LSTD)** algorithm computes coefficients by solving:

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} - \gamma \mathbf{P}_\pi \boldsymbol{\Phi}) \boldsymbol{\theta} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{r}_\pi,
$$

where $\boldsymbol{\Phi}$ is the matrix of basis function evaluations and $\boldsymbol{\Xi} = \text{diag}(\xi)$. We write this matrix equation for analysis purposes, but the actual algorithm does not compute it this way. For large state spaces, we cannot enumerate all states to form $\boldsymbol{\Phi}$ or explicitly represent the transition matrix $\mathbf{P}_\pi$. Instead, the practical algorithm accumulates sums from sampled transitions $(s, r, s')$, incrementally building the matrices $\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \boldsymbol{\Phi}$ and $\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{P}_\pi \boldsymbol{\Phi}$ without ever forming the full objects. The algorithm is derived from first principles through temporal difference learning, and the Galerkin perspective provides an interpretation of what it computes.

### LSTD as Projected Bellman Equation

To see what this equation means, let $\hat{v} = \boldsymbol{\Phi} \boldsymbol{\theta}$ be the solution. Expanding the parentheses:

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \boldsymbol{\Phi} \boldsymbol{\theta} - \gamma \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{P}_\pi \boldsymbol{\Phi} \boldsymbol{\theta} = \boldsymbol{\Phi}^\top \boldsymbol{\Xi} \mathbf{r}_\pi.
$$

Moving all terms to the left side and factoring out $\boldsymbol{\Phi}^\top \boldsymbol{\Xi}$:

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\boldsymbol{\Phi} \boldsymbol{\theta} - \gamma \mathbf{P}_\pi \boldsymbol{\Phi} \boldsymbol{\theta} - \mathbf{r}_\pi) = \mathbf{0}.
$$

Since $\hat{v} = \boldsymbol{\Phi} \boldsymbol{\theta}$ and the policy Bellman operator is $\BellmanPi \hat{v} = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi \hat{v}$, we can write:

$$
\boldsymbol{\Phi}^\top \boldsymbol{\Xi} (\hat{v} - \BellmanPi \hat{v}) = \mathbf{0}.
$$

Let $\boldsymbol{\varphi}_j$ denote the $j$-th column of $\boldsymbol{\Phi}$, which contains the evaluations of the $j$-th basis function at all states. The equation above says that for each $j$:

$$
\boldsymbol{\varphi}_j^\top \boldsymbol{\Xi} (\hat{v} - \BellmanPi \hat{v}) = 0.
$$

But $\boldsymbol{\varphi}_j^\top \boldsymbol{\Xi} (\hat{v} - \BellmanPi \hat{v})$ is exactly the $\xi$-weighted inner product $\langle \boldsymbol{\varphi}_j, \hat{v} - \BellmanPi \hat{v} \rangle_\xi$. So the residual $\hat{v} - \BellmanPi \hat{v}$ is orthogonal to every basis function, and therefore orthogonal to the entire subspace $\text{span}(\boldsymbol{\Phi})$. 

By definition, the orthogonal projection $\Proj y$ of a vector $y$ onto a subspace is the unique vector in that subspace such that $y - \Proj y$ is orthogonal to the subspace. Here, $\hat{v}$ lies in $\text{span}(\boldsymbol{\Phi})$ (since $\hat{v} = \boldsymbol{\Phi} \boldsymbol{\theta}$), and we have just shown that $\BellmanPi \hat{v} - \hat{v}$ is orthogonal to $\text{span}(\boldsymbol{\Phi})$. Therefore, $\hat{v} = \Proj \BellmanPi \hat{v}$, where $\Proj$ is orthogonal projection onto $\text{span}(\boldsymbol{\Phi})$ with respect to the $\xi$-weighted inner product:

$$
\langle u, v \rangle_\xi = u^\top \boldsymbol{\Xi} v, \qquad \|v\|_\xi = \sqrt{v^\top \boldsymbol{\Xi} v}, \qquad \Proj = \boldsymbol{\Phi}(\boldsymbol{\Phi}^\top \boldsymbol{\Xi} \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \boldsymbol{\Xi}.
$$

The weighting by $\xi$ is not arbitrary. Temporal difference learning performs stochastic updates using individual transitions: $\theta_{k+1} = \theta_k + \alpha_k (r + \gamma v_{\theta_k}(s') - v_{\theta_k}(s)) \nabla v_{\theta_k}(s)$, with states sampled from $\xi$. The ODE analysis of this stochastic process (Borkar-Meyn theory) shows convergence to a fixed point, which can be expressed in closed form as the $\xi$-weighted projected Bellman operator. LSTD is an algorithm that computes this analytical fixed point.

### Orthogonal Projection is Non-Expansive

Suppose $\xi$ is the steady-state distribution: $\xi^\top \mathbf{P}_\pi = \xi^\top$. Our goal is to establish that $\Proj \BellmanPi$ is a contraction in $\|\cdot\|_\xi$. If we can establish that $\Proj$ is non-expansive in this norm and that $\BellmanPi$ is a $\gamma$-contraction in $\|\cdot\|_\xi$, then their composition will be a $\gamma$-contraction:

$$
\|\Proj \BellmanPi v - \Proj \BellmanPi w\|_\xi \leq \|\BellmanPi v - \BellmanPi w\|_\xi \leq \gamma \|v - w\|_\xi.
$$


First, we establish that orthogonal projection is non-expansive. For any vector $v$, we can decompose $v = \Proj v + (v - \Proj v)$, where $(v - \Proj v)$ is orthogonal to the subspace $\text{span}(\boldsymbol{\Phi})$. By the Pythagorean theorem in the $\|\cdot\|_\xi$ inner product:

$$
\|v\|_\xi^2 = \|\Proj v\|_\xi^2 + \|v - \Proj v\|_\xi^2.
$$

Since $\|v - \Proj v\|_\xi^2 \geq 0$, we have:

$$
\|v\|_\xi^2 \geq \|\Proj v\|_\xi^2.
$$

Taking square roots of both sides (which preserves the inequality since both norms are non-negative):

$$
\|\Proj v\|_\xi \leq \|v\|_\xi.
$$

This holds for all $v$, so $\Proj$ is non-expansive in $\|\cdot\|_\xi$.

### Can $\BellmanPi$ be a Contraction in $\|\cdot\|_\xi$ ?

To show $\BellmanPi = r_\pi + \gamma \mathbf{P}_\pi$ is a $\gamma$-contraction, we need to verify:

$$
\|\BellmanPi v - \BellmanPi w\|_\xi = \|\gamma \mathbf{P}_\pi (v - w)\|_\xi = \gamma \|\mathbf{P}_\pi (v - w)\|_\xi.
$$

This will be at most $\gamma \|v - w\|_\xi$ if $\mathbf{P}_\pi$ is non-expansive, meaning $\|\mathbf{P}_\pi z\|_\xi \leq \|z\|_\xi$ for any vector $z$. So the key is to establish that $\mathbf{P}_\pi$ is non-expansive in $\|\cdot\|_\xi$. 

Consider the squared norm of $\mathbf{P}_\pi z$. By definition of the weighted norm:

$$
\|\mathbf{P}_\pi z\|_\xi^2 = \sum_s \xi(s) [(\mathbf{P}_\pi z)(s)]^2.
$$

The $s$-th component of $\mathbf{P}_\pi z$ is $(\mathbf{P}_\pi z)(s) = \sum_{s'} p(s'|s,\pi(s)) z(s')$. This is a weighted average of the values $z(s')$ with weights $p(s'|s,\pi(s))$ that sum to one. Therefore:

$$
\|\mathbf{P}_\pi z\|_\xi^2 = \sum_s \xi(s) \left[\sum_{s'} p(s'|s,\pi(s)) z(s')\right]^2.
$$

Since the function $x \mapsto x^2$ is convex, Jensen's inequality applied to the probability distribution $p(\cdot|s,\pi(s))$ gives:

$$
\left[\sum_{s'} p(s'|s,\pi(s)) z(s')\right]^2 \leq \sum_{s'} p(s'|s,\pi(s)) z(s')^2.
$$

Substituting this into the norm expression:

$$
\|\mathbf{P}_\pi z\|_\xi^2 \leq \sum_s \xi(s) \sum_{s'} p(s'|s,\pi(s)) z(s')^2 = \sum_{s'} z(s')^2 \sum_s \xi(s) p(s'|s,\pi(s)).
$$

The stationarity condition $\xi^\top \mathbf{P}_\pi = \xi^\top$ means $\sum_s \xi(s) p(s'|s,\pi(s)) = \xi(s')$ for all $s'$. Therefore:

$$
\|\mathbf{P}_\pi z\|_\xi^2 \leq \sum_{s'} z(s')^2 \xi(s') = \|z\|_\xi^2.
$$

Taking square roots, $\|\mathbf{P}_\pi z\|_\xi \leq \|z\|_\xi$, so $\mathbf{P}_\pi$ is non-expansive in $\|\cdot\|_\xi$. This makes $\BellmanPi = r_\pi + \gamma \mathbf{P}_\pi$ a $\gamma$-contraction in $\|\cdot\|_\xi$. Composing with the non-expansive projection:

$$
\|\Proj \BellmanPi v - \Proj \BellmanPi w\|_\xi \leq \|\BellmanPi v - \BellmanPi w\|_\xi \leq \gamma \|v - w\|_\xi.
$$

By Banach's fixed-point theorem, $\Proj \BellmanPi$ has a unique fixed point and iterates converge from any initialization.

### Interpretation: The On-Policy Condition

The result shows that convergence depends on matching the weighting to the operator. We cannot choose an arbitrary weighted $L^2$ norm and expect $\Proj \BellmanPi$ to be a contraction. Instead, the weighting $\xi$ must have a specific relationship with the transition matrix $\mathbf{P}_\pi$ in the operator $\BellmanPi$: namely, $\xi$ must be the stationary distribution of $\mathbf{P}_\pi$. This is what makes the weighted geometry compatible with the operator's structure. When this match holds, Jensen's inequality gives us non-expansiveness of $\mathbf{P}_\pi$ in the $\|\cdot\|_\xi$ norm, and the composition $\Proj \BellmanPi$ inherits the contraction property.

In reinforcement learning, this has a practical interpretation. When we learn by following policy $\pi$ and collecting transitions $(s, a, r, s')$, the states we visit are distributed according to the stationary distribution of $\pi$. This is **on-policy learning**. The LSTD algorithm uses data sampled from this distribution, which means the empirical weighting naturally matches the operator structure. Our analysis shows that the iterative algorithm $v_{k+1} = \Proj \BellmanPi v_k$ converges to the same fixed point that LSTD computes in closed form.

This is fundamentally different from the monotone approximator theory. There, we required structural properties of $\Proj$ itself (monotonicity, constant preservation) to guarantee that $\Proj$ preserves the sup-norm contraction property of $\Bellman$. Here, we place no such restriction on $\Proj$. Galerkin projection is not monotone. Instead, convergence depends on matching the norm to the operator. When $\xi$ does not match the stationary distribution, as in off-policy learning where data comes from a different behavior policy, the Jensen inequality argument breaks down. The operator $\mathbf{P}_\pi$ need not be non-expansive in $\|\cdot\|_\xi$, and $\Proj \BellmanPi$ may fail to contract. This explains divergence phenomena such as Baird's counterexample {cite}`Baird1995`.

### The Bellman Optimality Case

Can we extend this weighted $L^2$ analysis to the Bellman optimality operator $\Bellman v = \max_a [r_a + \gamma \mathbf{P}_a v]$? The answer is no, at least not with this approach. The obstacle appears at the Jensen inequality step. For policy evaluation, we had:

$$
\|\mathbf{P}_\pi z\|_\xi^2 = \sum_s \xi(s) \left[\sum_{s'} p(s'|s,\pi(s)) z(s')\right]^2.
$$

The inner term is a convex combination of the values $z(s')$, which allowed us to apply Jensen's inequality to the convex function $x \mapsto x^2$. For the optimal Bellman operator, we would need to bound:

$$
\left[\max_{a} \sum_{s'} p(s'|s,a) z(s')\right]^2.
$$

But the maximum of convex combinations is not itself a convex combination. It is a pointwise maximum. Jensen's inequality does not apply. We cannot conclude that $\max_a [\mathbf{P}_a z]$ is non-expansive in any weighted $L^2$ norm.

Is convergence of $\Proj \Bellman$ with Galerkin projection impossible, or merely difficult to prove? The situation is subtle. In practice, fitted Q-iteration and approximate value iteration with neural networks often work well, suggesting that some form of stability exists. But there are also well-documented divergence examples (e.g., Q-learning with linear function approximation can diverge). The theoretical picture remains incomplete. Some results exist for restricted function classes or under strong assumptions on the MDP structure, but no general convergence guarantee like the policy evaluation result is available. The interplay between the max operator, the projection, and the norm geometry is not well understood. This is an active area of research in reinforcement learning theory.

## Summary

This chapter developed weighted residual methods for solving functional equations like the Bellman equation. We approximate the value function using a finite basis, then impose conditions that make the residual orthogonal to chosen test functions. Different choices of test functions yield different methods: Galerkin tests against the basis itself, collocation tests at specific points, and least squares minimizes the residual norm. All reduce to the same computational pattern: generate Bellman targets, fit a function approximator, repeat.

Convergence depends on how the projection interacts with the Bellman operator. For monotone projections (piecewise linear interpolation, state aggregation), the composition $\Proj \Bellman$ inherits the contraction property and iteration converges. For non-monotone projections like Galerkin, convergence requires matching the weighting to the stationary distribution, which holds in on-policy settings. The Bellman optimality case remains theoretically incomplete.

Throughout this chapter, we assumed access to the transition model: computing $\Bellman v(s)$ requires summing over all next states weighted by transition probabilities. In practice, we often have only a simulator or observed trajectories, not an explicit model. The next chapter addresses this gap. Monte Carlo methods estimate expectations from sampled transitions, replacing exact Bellman operator evaluations with sample averages. This connects the projection framework developed here to the simulation-based algorithms used in reinforcement learning.
