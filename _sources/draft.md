#### Newton-Kantorovich Method

The Newton-Kantorovich method is a generalization of Newton's method from finite dimensional vector spaces to infinite dimensional function spaces: rather than iterating in the space of vectors, we are iterating in the space of functions. Just as in the finite-dimensional counterpart, the idea is to improve the rate of convergence of our method by taking an "educated guess" on where to move next using a linearization of our operator at the current point. Now the concept of linearization, which is synonymous with derivative, will also require a generalization. Here we are in essence trying to quantify how the output of the operator $T$ -- a function -- varies as we perturb its input -- also a function. The right generalization here is that of the Fréchet derivative.

The Fréchet derivative is a generalization of the derivative to Banach spaces. For an operator $T: X \to Y$ between Banach spaces $X$ and $Y$, the Fréchet derivative at a point $x \in X$, denoted $T'(x)$, is a bounded linear operator from $X$ to $Y$ such that:

$$
\lim_{h \to 0} \frac{\|T(x + h) - T(x) - T'(x)h\|_Y}{\|h\|_X} = 0
$$

where $\|\cdot\|_X$ and $\|\cdot\|_Y$ are the norms in $X$ and $Y$ respectively. In other words, $T'(x)$ is the best linear approximation of $T$ near $x$.

But apart from those mathematical technicalities, Newton-Kantorovich has in essence the same structure as that of the original Newton's method. That is, it applies the following sequence of steps:

1. **Linearize the Operator**:
   Given an approximation \( x_n \), we consider the Fréchet derivative of \( T \), denoted by \( T'(x_n) \). This derivative is a linear operator that provides a local approximation of \( T \) near \( x_n \).

2. **Set Up the Newton Step**:
   The method then solves the linearized equation for a correction \( h_n \):
   $$
   T'(x_n) h_n = T(x_n) - x_n.
   $$
   This equation represents a linear system where \( h_n \) is chosen to minimize the difference between \( x_n \) and \( T(x_n) \) with respect to the operator's local behavior.

3. **Update the Solution**:
   The new approximation \( x_{n+1} \) is then given by:
   $$
   x_{n+1} = x_n - h_n.
   $$
   This correction step refines \( x_n \), bringing it closer to the true solution.

4. **Repeat Until Convergence**:
   We repeat the linearization and update steps until the solution \( x_n \) converges to the desired tolerance, which can be verified by checking that \( \|T(x_n) - x_n\| \) is sufficiently small, or by monitoring the norm \( \|x_{n+1} - x_n\| \).

The convergence of Newton-Kantorovich does not hinge on \( T \) being a contraction over the entire domain -- as it could be the case for successive approximation. The convergence properties of the Newton-Kantorovich method are as follows:

1. **Local Convergence**: Under mild conditions (e.g., $T$ is Fréchet differentiable and $T'(x)$ is invertible near the solution), the method converges locally. This means that if the initial guess is sufficiently close to the true solution, the method will converge.

2. **Global Convergence**: Global convergence is not guaranteed in general. However, under stronger conditions (e.g., $T$ is analytic and satisfies certain bounds), the method can converge globally.

3. **Rate of Convergence**: When the method converges, it typically exhibits quadratic convergence. This means that the error at each step is proportional to the square of the error at the previous step:

   $$
   \|x_{n+1} - x^*\| \leq C\|x_n - x^*\|^2
   $$

   where $x^*$ is the true solution and $C$ is some constant. This quadratic convergence is significantly faster than the linear convergence typically seen in methods like successive approximation.

Under appropriate conditions (e.g., if \( T \) is sufficiently smooth), it exhibits quadratic convergence, which can be significantly faster than the linear convergence of the successive approximation method. This rapid convergence makes the Newton-Kantorovich method particularly powerful for solving nonlinear problems in function spaces, despite the increased complexity of each iteration compared to simpler methods.