\documentclass{article}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{geometry}
\usepackage{mathrsfs}
\geometry{margin=1in}

\title{Projection Methods for Functional Equations \\ \large From Kenneth Judd's \textit{Numerical Methods in Economics}}
\author{}
\date{}

\begin{document}

\maketitle

\section*{General Projection Approach}

Suppose that we want a solution to the operator equation
$$\mathscr{N}(f)=0,$$
where $\mathscr{N}: B_1 \rightarrow B_2$, with $B_1$ and $B_2$ complete normed vector spaces of functions $f: D \subset \mathbb{R}^n \rightarrow \mathbb{R}^m$, and where $\mathscr{N}$ is a continuous map. In our simple ordinary differential equation example (11.1.1), $D=[0, T]$, $f: D \rightarrow \mathbb{R}$, $\mathscr{N}=d / dt-I$ where $I$ is the identity operator, $B_1$ is the space of $C^1$ functions, and $B_2$ is the space of $C^0$ functions. In this example, $B_2$ contained $B_1$, and both are contained in the space of measurable functions. More generally, $f$ is a list of functions in the definition of equilibrium, such as decision rules, price functions, value functions, and conditional expectations functions, and the $\mathscr{N}$ operator expresses equilibrium conditions such as market clearing, Euler equations, Bellman and HJB equations, and rational expectations.

Since we are focused on describing the computational method, we will specify the topological details only to the extent implicit in the computational details. For example, we do not say exactly what norms and inner products are being implicitly used in the ODE and PDE examples in the previous sections. While these topological details are important, they lie far beyond the scope of this book. This may make some readers uncomfortable, particularly when we note that the topological aspects of many of our applications are not well-understood. Some comfort can be taken in the fact that we always check the validity of our solutions, and those checks may keep us from accepting projection method solutions in cases where the underlying functional analytic structure does not support the use of such methods. The reader can also see Zeidler (1986) and Krasnosel'ski\u{\i} and Zabreiko (1984) for serious discussions of those issues, and is encouraged to apply the methods there to economic problems.

The first step is to decide how to represent approximate solutions to $\mathscr{N}(f)=0$. One general way is to assume that our approximation, $\hat{f}$, is built up as a linear\footnote{Nonlinear combinations are also possible, but we stay with linear combinations here, since linear approximation theory is a much more developed theory than nonlinear approximation theory.} combination of simple functions from $B_1$. We will also need concepts of when two functions are close or far apart. Therefore, the first step is to choose bases and concepts of distance:

\paragraph{STEP 1} Choose a basis over $B_1$, $\Phi=\left\{\varphi_i\right\}_{i=1}^{\infty}$ and a norm, $\|\cdot\|$. Similarly choose a basis over $B_2$, $\Psi=\left\{\psi_i\right\}_{i=1}^{\infty}$ and an inner product, $\langle\cdot, \cdot\rangle_2$ over $B_2$.

When $B_1$ and $B_2$ are subsets of another space, we will often use a basis of the larger space as the basis of $B_1$ and an inner product norm of the larger space as the norm on $B_2$. Our approximation to the solution of $\mathscr{N}(f)=0$ will be denoted $\hat{f}$; we next decide how many of these basis elements we will use.

\paragraph{STEP 2} Choose a degree of approximation, $n$, and define $\hat{f} \equiv \sum_{i=1}^n a_i \varphi_i(x)$.
\footnote{The convention is that the $\varphi_i$ increase in "complexity" and "nonlinearity" as $i$ increases, and that the first $n$ elements are used. In the case of standard families of orthogonal polynomials, $\varphi_i$ is the degree $i-1$ polynomial.}

Step 1 lays down the structure of our approximation, and step 2 fixes the flexibility of the approximation. Once we have made these basic decisions, we begin our search for an approximate solution to (11.3.1). Since the only unknown part of the approximation is the vector $a$, we have reduced the original infinite-dimensional problem to a finite-dimensional one. If our diagnostic tests leave us dissatisfied with that approximation, we can return to step 2 and increase $n$ in hopes of getting an improved approximation. If that fails, we can return to step 1 and begin again with a different basis.

Since the true solution $f$ satisfies $\mathscr{N}(f)=0$, we will choose as our approximation some $\hat{f}$ that makes $\mathscr{N}(\hat{f})$ nearly equal to the zero function, where by near we refer to properties defined by the norm in $B_2,\|\cdot\|_2$, which corresponds to the inner product $\langle\cdot, \cdot\rangle_2$. Since $\hat{f}$ is parameterized by $a$, the problem reduces to finding an $a$ which makes $\mathscr{N}(\hat{f})$ nearly zero. In many cases computing $\mathscr{N}(\hat{f})$ is also challenging, such as when $\mathscr{N}(f)$ involves integration of $f$; in those cases we need to approximate the $\mathscr{N}$ operator.

\paragraph{STEP 3} Construct a computable approximation, $\hat{\mathscr{N}}$, to $\mathscr{N}$, and define the residual function
$$
R(x ; a) \equiv(\hat{\mathscr{N}}(\hat{f}(\cdot ; a))(x) .
$$
Steps 2 and 3 transform an operation in an infinite-dimensional space into a computable finite-dimensional one. We need next to specify our notion of $\hat{f}$ nearly solving $\mathscr{N}(f)=0$.

\paragraph{STEP 4} Either compute the norm of $R(\cdot ; a),\|R(\cdot ; a)\| \equiv\langle R(\cdot ; a), R(\cdot ; a)\rangle$, or choose a collection of $l$ test functions in $B_2, p_i: D \rightarrow \mathbb{R}^m, i=1, \ldots, l$, and for each guess of $a$ compute the $l$ projections, $P_i(\cdot) \equiv\left\langle R(\cdot ; a), p_i(\cdot)\right\rangle$.

Step 4 creates the projections we will use. The choices made in step 4 generally give the projection method its name. Projection methods are also called "weighted residual methods," since the criteria in step 4 weigh the residuals. Once we have chosen our criterion, we can determine the value of the unknown coefficients, $a$.

\paragraph{STEP 5} Find $a \in \mathbb{R}^n$ that either minimizes $\|R(\cdot ; a)\|$ or solves $P(a)=0$.
When we have our solution, $\hat{f}$, we are not done. It is only a candidate solution, and we must test the quality of our solution before accepting it.

\paragraph{STEP 6} Verify the quality of the candidate solution by approximating $\mathscr{N}(\hat{f})$.
We accomplish step 6 by computing the norm $\|\mathscr{N}(\hat{f})\|$ and/or projections of $\mathscr{N}(\hat{f})$ against test functions not used in step 4. If $\mathscr{N}$ must be approximated, we should, if feasible, use a better approximation in step 6 than that constructed in step 3. Ideally all of these quantities will be zero; in practice, we choose target quantities for these diagnostics that imply that the deviations from zero are not economically significant.

This general algorithm breaks the numerical problem into several distinct steps. It points out the many distinct techniques of numerical analysis which are important. First, in steps 1 and 2 we choose the finite-dimensional space wherein we look for approximate solutions, hoping that within this set there is something "close" to the real solution. These steps require us to think seriously about approximation theory methods. Second, step 4 will involve numerical integration if we cannot explicitly compute the integrals that define the projections, and step 3 frequently involves numerical integration in economics applications. Third, step 5 is a third distinct numerical problem, involving the solution of a nonlinear set of simultaneous equations or the solution of a minimization problem. We will now consider each of these numerical problems in isolation.

\subsection*{Choice of Basis and Approximation Degree}
There are many criteria that the basis and inner product should satisfy. The full basis $\Phi$ for the space of candidate solutions should be "rich," flexible enough to approximate any function relevant to the problem. The best choice of $n$ cannot be determined a priori. Generally, the only correct choice is $n=\infty$. If the choice of the basis is good, then larger $n$ will yield better approximations. We are most interested, however, in the smallest $n$ that yields an acceptable approximation. We initially begin with small $n$ and increase $n$ until some diagnostic indicates little is gained by continuing. Computational considerations also play a role in choosing a basis. The $\varphi_i$ should be simple to compute, and all be similar in size to avoid scaling problems.

Of course, the number of basis elements needed will depend greatly on the basis being used; using a basis that is well-suited to a problem can greatly improve performance. Approximation theory, discussed in chapter 6, can be used to evaluate alternative bases because ultimately we are trying to approximate the solution $f$ with a finite combination of simple known functions. The basis elements should look something like the solution so that only a few elements can give a good approximation. While asymptotic results such as the Stone-Weierstrass theorem may lull one into accepting polynomial approximations, practical success requires a basis where only a few elements will do the job. The individual terms should also be "different"; ideally they should be orthogonal with respect to the inner product $\langle\cdot, \cdot\rangle$. The reasons are essentially the same as why one wants uncorrelated explanatory variables in a regression. Nonorthogonal bases will reduce numerical accuracy just as multicollinear regressors enlarge confidence intervals. Algorithms that solve for the unknown coefficients $a$ solve several linear systems of equations, and the accuracy of these solutions depends on the rows and columns not being collinear. Orthogonal bases will help avoid ill-conditioned matrices in these intermediate steps.

From chapter 6 we know that there are several possible bases. First, let us consider the ordinary polynomials, $\left\{1, x, x^2, x^3, \ldots\right\}$. If $B_1$ is the set of bounded measurable functions on a compact set then the Stone-Weierstrass theorem assures us of their completeness in the $L_1$ norm. However, they may not be a good choice since they are too similar. For example, they are all monotonically increasing and positive on $\mathbb{R}^{+}$, and they will not be orthogonal in any natural inner product on $\mathbb{R}^{+}$. They will also vary a great deal in size over most intervals $D$. The ordinary polynomials will sometimes be adequate, as they were in our simple examples, because we needed few terms to get a good solution.

These considerations do not mean that we cannot use ordinary polynomials, just that it is preferable to use polynomial bases that are orthonormal with respect to the inner product. A generally useful choice are systems of Chebyshev polynomials, which were discussed in chapter 6. Nonpolynomial alternatives include various sequences of trigonometric and exponential functions. The choice depends on the range, $D$, computational demands, and the expected shape of a solution. In physics, trigonometric bases such as $\{1, \sin x, \sin 2 x, \sin 3 x, \ldots\}$ are often used, since solutions are often periodic, allowing for Fourier series techniques. In economic problems, however, it is better to use nontrigonometric bases, since solutions are generally not periodic and periodic approximations to nonperiodic functions require many terms.

We can use the full array of approximation methods discussed in chapter 6. Some families of projection methods are known by their method of approximation. Spectral methods use bases where each element is nonzero almost everywhere, such as in trigonometric bases and orthogonal polynomials. Finite element methods use bases where each element has small support, as discussed in section 6.13.

Most interesting problems in economics involve more than one state variable-physical versus human capital, capital stocks of oligopolistic competitors, wealth distribution across investor groups, and so on. The tensor product methods discussed in Chapter 6 build up multidimensional basis functions from simple one-dimensional basis functions. The curse of dimensionality, a problem that arises with tensor product bases, can be avoided by using the complete polynomials as a basis instead.

We are not limited to the conventional approaches described in chapter 6. If we have some reason to believe that a solution will look like some nonconventional functions, then we should try them. We may want to orthogonalize the family, and we may need to develop the corresponding Gaussian quadrature formulas, but all that is just a straightforward application of the methods discussed in chapters 6 and 7. This may be quite important in multidimensional problems because we will need to economize on basis elements. In chapter 15 we will discuss formal ways to generate good problem-specific bases. Even though we will focus here on using standard approximation methods, the ideas of projection methods generalize directly to more idiosyncratic choices.

\subsection*{Choice of Projection Conditions}
As we have seen in our examples, projection techniques include a variety of special methods. In general, we specify some inner product, $\langle\cdot, \cdot\rangle$, of $B_2$, and use $\langle\cdots\rangle$ to measure the "size" of the residual function, $R$, or its projection against the test functions. We can use inner products of the form
$$
\langle f(x), g(x)\rangle \equiv \int_D f(x) g(x) w(x) d x
$$
for some weighting function $w(x)$, but there is no reason why we are limited to them. In choosing the norm, one should consider exactly what kind of error should be small and find a norm that will be sensitive to the important errors. There are several ways to proceed.

The general least squares projection method computes the $L^2$ norm of the residual function, namely $\langle R(x ; a), R(x ; a)\rangle$, and chooses $a$ so as to minimize the "sum of squared residuals":
$$
\min _a\langle R(x ; a), R(x ; a)\rangle .
$$
We have thereby reduced the problem of solving a functional equation to solving a nonlinear minimization problem in $\mathbb{R}^n$, a more tractable problem. Of course, the standard difficulties will arise. For example, there may be local minima which are not global solutions. However, there is no reason for these problems to arise more often here than in any other context, such as maximum likelihood estimation, where minimization problems are solved numerically.

The least squares method is a direct implementation of the idea to make small the error of the approximation. In general, one could develop alternative implementations by using different norms. However, most projection techniques find a goodfitting approximation in less direct fashions. For these techniques the basic idea is that the true solution would have a zero residual error function; in particular, its projection in all directions is zero. Therefore one way to find the $n$ components of $a$ is to fix $n$ projections and choose $a$ so that the projection of the resulting residual function in each of those $n$ directions is zero. Formally these methods find $a$ such that $\left\langle R, p_i\right\rangle=0$ for some specified collection of test functions, $p_i$. Different choices of the $p_i$ defines different implementations of the projection method.

It is clear that the least squares and alternative implementations of projection ideas are similar since one way to solve the least squares approach is to solve the nonlinear set of equations generated by its first-order conditions, $\left\langle R, \partial R / \partial a_i\right\rangle=0$. Seeing the least squares method expressed as a system of projection equations gives us some indication why other methods may be better. The projection directions in the least squares case, the gradients of the residual function, could be highly correlated. Furthermore the projection directions depend on the guess for $a$. This lack of control over the implicit projection directions is not a good feature. Also in economic problems we may have a preference for approximations that have zero projections in certain directions, such as the average error in an Euler equation. Many of the alternative techniques will naturally include that condition.

One such alternative technique is the Galerkin method, also known as the BubnovGalerkin or Galerkin-Petrov method. In the Galerkin method we use the first $n$ elements of the basis for the projection directions, where we are making the weak assumption that $\Phi$, our basis of $B_1$, lies in $B_2$. Therefore $a$ is chosen to solve the following set of equations:
$$
P_i(a) \equiv\left\langle R(x ; a), \varphi_i(x)\right\rangle=0, \quad i=1, \ldots, n
$$
Notice that here we have reduced the problem of solving a differential equation to one of solving a set of nonlinear equations. In some cases the Galerkin projection equations are the first-order conditions to some minimization problem, as is often the case in linear problems from physics. When we have such an equivalence, the Galerkin method is also called the Rayleigh-Ritz method. This is not as likely to happen in economics problems because of nonlinearities.

The method of moments, subdomain, and collocation procedures can be applied to the general setting. If $D \subset \mathbb{R}$, then the method of moments chooses the first $n$ polynomials for the projection directions; that is, we find $a$ that solves the system
$$
P_i(a) \equiv\left\langle R(x ; a), x^{i-1}\right\rangle=0, \quad i=1, \ldots, n
$$
If $D$ is of higher dimension, then we project $R$ against a sufficient number of loworder multivariate monomials. In the subdomain method the idea is to find an approximation that is good on average on a collection of subsets that cover the whole domain. More specifically, we choose $a$ so that
$$
P_i(a) \equiv\left\langle R(x ; a), I_{D_i}\right\rangle=0, \quad i=1, \ldots, n
$$
where $\left\{D_i\right\}_{i=1}^n$ is a sequence of intervals covering $D$, and $I_{D_i}$ is the indicator function for $D_i$.

The collocation method chooses $a$ so that the functional equation holds exactly at $n$ fixed points. That is, we choose $a$ to solve
$$
R\left(x_i ; a\right)=0, \quad i=1, \ldots, n
$$
where $\left\{x_i\right\}_{i=1}^n$ are $n$ fixed points from $D$. This is a special case of the projection approach, since $R\left(x_i ; a\right)$ equals $\left\langle R(x ; a), \delta\left(x-x_i\right)\right\rangle$, the projection of $R(x ; a)$ against the Dirac delta function at $x_i$.

Orthogonal collocation is the method where the $x_i$ are the $n$ zeros of the $n$th orthogonal polynomial basis element and the basis elements are orthogonal with respect to the inner product. It is a particularly powerful application of projection ideas when used with a Chebyshev polynomial basis. This is not a surprise in light of the Chebyshev interpolation theorem. Suppose that $D=[-1,1]$ and $R\left(x_i ; a\right)=0, i=1, \ldots, n$, where the $x_i$ are the $n$ zeros of $T_n$. As long as $R(x; a)$ is smooth in $x$, the Chebyshev interpolation theorem says that these zero conditions force $R(x ; a)$ to be close to zero for all $x \in[-1,1]$. The optimality of Chebyshev interpolation also says that if one is going to use collocation, these are the best possible points to use. Even after absorbing these considerations, it is not certain that even Chebyshev collocation is a reliable method. We will see below that its performance is surprisingly good.

Collocation can be used for bases other than orthogonal polynomials. Spline collocation methods use spline bases. The collocation points could be the spline nodes themselves or some other set of points, such as the midpoints of the spline mesh. The key objective is keeping the Jacobian of the collocation equation system well-conditioned.

\subsection*{Evaluation of Projections}
The meat of the problem is step 4, where the major computational task is the computation of those projections. The collocation method is fastest in this regard because it only uses the value of $R$ at $n$ points. More generally, the projections will involve integration. In some cases one may be able to explicitly perform the integration. This is generally possible for linear problems, and possible for special nonlinear problems. However, our experience with the economic applications below is that this will generally be impossible for nonlinear economic problems. We instead need to use quadrature techniques to compute the integrals associated with the evaluation of $\langle\cdot, \cdot\rangle$. A typical quadrature formula approximates $\int_a^b f(x) w(x) d x$ with a finite sum $\sum_{i=1}^n \omega_i f\left(x_i\right)$ where the $x_i$ are the quadrature nodes and the $\omega_i$ are the weights. Since these formulas also evaluate $R$ at just a finite number of points, quadrature-based projection techniques are essentially weighted collocation methods. The advantage of quadrature formulas is that information at more points is used to compute a more accurate approximation of the projections.

\subsection*{Finding the Solution}
Step 5 uses either a minimization algorithm or a nonlinear algebraic equation solver. If the system $P(a)=0$ is overidentified or if we are minimizing $\|R(\cdot ; a)\|$, we may invoke a nonlinear least squares algorithm. The nonlinear equations associated with Galerkin and other inner product methods can be solved by the variety of methods discussed in chapter 5. While fixed-point iteration appears to be popular in economics, Newton's method and its refinements have often been successful. Homotopy methods can also be used if one has no good initial guesses.

\subsection*{Initial Guesses}
Good initial guesses are important since projection methods involve either a system of nonlinear equations or optimizing a nonlinear, possibly multimodal, objective. Fortunately this is generally not a big problem. Often there are degenerate cases for which we can find the solution, which in turn will be a good guess for the problem we want to solve. The perturbation methods discussed in chapters 13 and 14 often generate good initial guesses. In some problems there are problem-specific ways of generating good initial guesses.

There is one general approach which is often useful. The least squares approach may not be a good one to use for high-quality approximations. However, it may yield low-quality approximations relatively quickly, and, since the least squares method is an optimization method, convergence to a local extrema is ensured even if one has no good initial guess. Furthermore, by adding terms to the least squares objective, one can impose sensible restrictions on the coefficients to eliminate economically nonsensical extrema. These facts motivate a two-stage approach. First, one uses a least squares approach with a loose convergence criterion to quickly compute a low-quality approximation. Second, one uses this approximation as the initial guess for a projection method attempting to compute a higher-order approximation. With some luck the least squares solution will be a good initial guess for the second computation. If it is difficult to find good initial guesses, then one can use homotopy methods that are globally convergent.

\subsection*{Coordination among Steps 1-5}
We now see what is needed for efficiency. The key is to choose elements for the separate steps that work well together. We need basis functions that are easy to evaluate because they will be frequently evaluated. Any integration in steps 3 and 4 must be accurate but fast. Therefore we should use quadrature formulas that work well with the basis. The nonlinear equation solver in step 5 needs to be efficient and should be able to use all the information arising from step 4 calculations. Step 5 will typically use gradient information about the integrals of step 4. It is therefore important to do those gradient calculations quickly, doing them analytically when practical.

A particularly important interaction is that between the choice of a basis and the solution of the nonlinear problem in $\mathbb{R}^n$. Most methods for solving the system $P(a)=0$ will use its Jacobian, $P_a(a)$. If this matrix is nearly singular near the solution, accuracy will be poor due to round-off error and convergence will be slow. Choosing an orthogonal basis, or nearly orthogonal basis (as is the case with B splines), will substantially reduce the likelihood of a poorly conditioned Jacobian, even in nonlinear problems.

Most methods used in numerical analysis of economic models fall within the general description for projection methods. We will see these connections below when we compare how various methods attack a common problem. The key fact is that the methods differ in their choices of basis, fitting criterion, and integration technique.

\subsection*{Evaluating a Solution}
As with operator equation methods in general, the projection algorithm does not automatically evaluate the quality of the candidate approximate solution. One of the advantages of the projection approach is the ease with which one can do the desired evaluation. The key observation is that we typically use an approximation to $\mathscr{N}, \hat{\mathscr{N}}$, when searching for the approximate solution, $\hat{f}$. For example, we use numerical integration methods to compute conditional expectations and to compute projections. To economize on computer time, we use the least amount of information possible to compute $\hat{f}$.

This is a risky but acceptable strategy, since we accept $\hat{f}$ only after we strenuously test the candidate $\hat{f}$. Therefore we use better quadrature rules here to evaluate $\hat{f}$, and we check if $0=(\mathscr{N}(\hat{f}))(x)$ at many points that were not used in the derivation of $\hat{f}$. We also use a finite number of test functions in constructing $\hat{f}$, leaving an infinite number of test functions that we can use to evaluate $\hat{f}$. While this discussion is abstract here, the actual implementation will be quite intuitive in actual applications.

\subsection*{Existence Problems}
Projection methods are useful ways to transform infinite-dimensional problems into finite-dimensional problems. However, these transformations present problems that we have not faced in previous methods-existence. In previous chapters we investigated methods that were more similar to the problem being analyzed. For example, in chapter 5 we examined nonlinear equations, where there is a solution to the numerical problem if and only if there is a solution to the original, pure problem. This can be different here. Sometimes the finite-dimensional problem generated by a projection method will not have a solution even when the original, pure problem does have a solution. Therefore, if one is having difficulty solving a projection equation system for a particular basis and particular $n$, the problem may go away just by trying another $n$ or another basis. For well-behaved problems, choosing a sufficiently large $n$ will work.

One way to ensure existence of a solution but gain some of the advantages of the projection methods is to construct a least squares objective from the projections. For example, in the case of the Galerkin method, one could solve the least squares problem
$$
\min _a \sum_{i=1}^n\left\langle R(x ; a), \varphi_i(x)\right\rangle^2 .
$$
One could use an overidentification approach and solve the problem
$$
\min _a \sum_{i=1}^m\left\langle R(x ; a), \varphi_i(x)\right\rangle^2
$$
for some $m>n$. These combinations of least squares and projection ideas are compromises. Existence of the finite-dimensional approximation is assured as long as the objectives are continuous, and optimization methods can be reliably used to find solutions.

In beginning a numerical analysis of a model, it is important to maintain flexibility as to which method to use. Since the projection methods are so similar, it is easy to change from one to another. Experimentation with several methods is the only way to find out which will be best.

\subsection*{Consistency}
When using numerical procedures, it is desirable to know something concerning the error of the solution. As we discussed in chapter 2, an important focus of theoretical numerical analysis is deriving error bounds and proving that methods are asymptotically valid. For example, we would like to know that the errors of a projection method go to zero as we enlarge the basis; this property is called consistency. There has been little work on proving that the algorithms used by economists are asymptotically valid.

The absence of convergence theorems does not invalidate the projection approach. The compute and verify approach will help avoid bad approximations. Even if we had a consistent method, we would have to develop a stopping criterion for $n$, the degree of the approximation, which itself would be a verification procedure. This is in keeping with our philosophy that a convergence theorem is not necessary for a procedure to be useful nor do convergence theorems make any particular result more reliable. In practice, we must use a method which stops at some finite point, and any candidate solution produced by any method must be tested before it is accepted.

\newpage
\section*{Numerical Dynamic Programming}
\subsection*{12.7 Continuous Methods for Continuous-State Problems}

Most of the methods examined so far either assume that there were only a finite number of states, or approximate a continuous state with a finite set of values. These methods are reliable in that they will solve the problem, or will approach the solution as the discretization is made finer. However, discretization becomes impractical as one moves to larger problems. In this section we introduce a parametric approach due to Bellman et al. (1963).

We don't use discretization to solve linear-quadratic problems because we know the functional form of the solution. This allows us to focus on finding the appropriate coefficients. Linear-quadratic problems do not suffer a curse of dimensionality since the number of unknown coefficients do not grow exponentially in the dimension. Even though most dynamic programming problems do not have a useful functional form, we can still use the functional form approach by exploiting approximation ideas from chapter 6.

Recall the basic Bellman equation:

$$
V(x)=\max _{u \in D(x)} \pi(u, x)+\beta E\left\{V\left(x^{+}\right) \mid x, u\right\} \equiv(T V)(x) .
$$


The unknown here is the function $V$. Discretization methods essentially approximate $V$ with a step function, since it implicitly treats any state between $x_i$ and $x_{i+1}$ as either $x_i$ or $x_{i-1}$. Chapter 6 presented better methods to approximate continuous functions. We apply those ideas here.

We assume that the payoff, motion, and value functions are all continuous functions of their arguments. The basic idea is that it should be better to approximate the continuous-value function with continuous functions and put no restrictions on the states and controls other than those mandated by the problem. Since the computer cannot model the entire space of continuous functions, we focus on a finitely parameterizable collection of functions,

$$
V(x)=\hat{V}(x ; a) .
$$

The functional form $\hat{V}$ may be a linear combination of polynomials, or it may represent a rational function or neural network representation with parameters $a \in R^m$, or it may be some other parameterization specially designed for the problem.

Once we fix the functional form, we focus on finding coefficients $a \in R^m$ such that $\hat{V}(x ; a)$ "approximately" satisfies the Bellman equation. Solving the Bellman equation, (12.7.1), means finding the fixed point of $T$, but that is the pure mathematical problem. The basic task for a numerical procedure is to replace $T$, an operator mapping continuous functions to continuous functions, with a finite-dimensional approximation, $\hat{T}$, which maps the set of functions of the form $\hat{V}$ into itself. If done properly, the fixed point of $\dot{T}$ should be close to the fixed point of $T$.

The construction of $\hat{T}$ relies on three critical steps. First, we choose some parameterization scheme $\hat{V}(x ; a)$ with $a \in R^m$, and $n$ points in the state space,

$$
X=\left\{x_1, x_2, \ldots, x_n\right\},
$$

where $n \geq m$. Second, we then evaluate $v_i=(T \hat{V})\left(x_i\right)$ at each $x_i \in X$ : we refer to this as the maximization step. The maximization step gives us values $v_i$ which are points on the function $T \hat{V}$. We next use the information about $T \hat{V}$ contained in the $v_i, i=1, \ldots, m$, to find an $a \in R^m$ such that $\dot{V}(x: a)$ best fits the $\left(v_i, x_i\right), i=1 \ldots, n$, data; we call this the fitting step. The fitting step can be an unweighted nonlinear least squares procedure as in

$$
\min _{a \in R^m} \sum_{i=1}^n\left(\hat{V}\left(x_i ; a\right)-v_i\right)^2
$$

or any other appropriate approximation scheme. This fitting step produces a new value function defined on all states $x$. The parameteric dynamic programming algorithm is outlined in algorithm 12.5.

\subsection*{Algorithm 12.5 Parametric Dynamic Programming with Value Function Iteration}
Objective: Solve the Bellman equation, (12.7.1),
Initialization. Choose functional form for $\dot{V}(x ; a)$, and choose the approximation grid, $X=\left\{x_1, \ldots, x_n\right\}$. Make initial guess $\hat{V}\left(x: a^0\right)$, and choose stopping criterion $\varepsilon>0$.
Step 1. Maximization step: Compute $y_j=\left(T \dot{V}\left(\because a^i\right)\right)\left(x_j\right)$ for $x_j \in X$.
Step 2. Fitting step: Using the appropriate approximation method. compute the $a^{i+1} \in R^m$ such that $\hat{V}\left(x ; a^{i+1}\right)$ approximates the $\left(v_i, x_i\right)$ data.
Step 3. If $\left\|\hat{V}\left(x ; a^i\right)-\hat{V}\left(x ; a^{i+1}\right)\right\|<\varepsilon$, stop; else go to step 1.

Steps 1 and 2 in algorithm 12.5 constitute a mapping $\hat{T}$ taking $\hat{V}$, corresponding to a parameter vector $a$, to another function $\hat{T} \hat{V}$ corresponding to another coefficient vector, $a^{\prime} . \dot{T}$ is therefore a mapping in the space of coefficients, a subspace of $R^m$.

Algorithm 12.5 presents the general idea; we now examine the numerical details. Rewrite (12.7.1) as

$$
V(x)=\max _{u \in D(x)} \pi(u, x)+\beta \int V\left(x^{+}\right) d F\left(x^{+} \mid x, u\right),
$$

where $F\left(x^{+} \mid x, u\right)$ is the distribution of the future state $x^{+}$conditional on the current state and control. The maximization steps compute

$$
v_j=\max _{u \in D\left(x_j\right)} \pi\left(u, x_j\right)+\beta \int \hat{V}\left(x^{+} ; a\right) d F\left(x^{+} \mid x_j, u\right), \quad x_j \in X .
$$


With the $v_j$ values for the grid $X$, we next compute a new value function approximation $\hat{V}(x ; a)$ which fits the new $v_j$ values at the $x_j \in X$.

These expressions make clear the three kinds of problems we need to handle. The first type of numerical problem is evaluating the integral $\int \hat{V}\left(x^{+}\right) d F\left(x^{+} \mid x, u\right)$. This would usually require numerical quadrature. If the integrand is smooth in $x^{+}$for fixed $x$ and $u$, a Gaussian approach is suggested. Less well-behaved integrals would require low-order Newton-Cotes formulas. High-dimensional integrands would require a monomial, Monte Carlo, or quasi-Monte Carlo method.

The second type of numerical problem appearing in (12.7.4) is the optimization problem, (12.7.5). If the objective in (12.7.5) is smooth in $u$, we could use faster methods such as Newton's method. It is important to choose an approximation scheme $\hat{V}$ which preserves any smoothness properties of the problem.

Third, given the $v_i$ estimates of the $(T \hat{V})\left(x_i\right)$ values, we need to compute the new coefficients in $\hat{V}(x ; a)$. The appropriate approximation procedure depends on the nature of $V$. If we expect $V$ to be a smooth function, then orthogonal polynomial procedures may be appropriate. Otherwise, splines may be advisable. We will see below that these considerations are particularly important here.

There is substantial interaction across the three problems. Smooth interpolation schemes allow us to use Newton's method in the maximization step. They also make it easier to evaluate the integral in (12.7.5). Since the integral in (12.7.5) is costly to evaluate, we may want to use different rules when computing the gradients and Hessian, using the observation that low-quality gradient and Hessian approximations often suffice.

While algorithm 12.5 looks sensible, there are several questions concerning its value. First, convergence is not assured since $\bar{T}$ may not be a contraction map; in

fact, we will see that it may be quite ill-behaved. The key detail is the choice of the approximation scheme incorporated in $\hat{V}(x ; a)$ and the grid $X$. The discussion below will focus on the behavior of this algorithm for various choices of these elements.

We should emphasize that we can still use some of the techniques developed for the finite-state case. Algorithm 12.5 presents only value iteration, but we could also implement policy iteration. Many other procedures are not so easily adapted for this approach. The Gauss-Scidel ideas, particularly the upwind methods, do not have obvious counterparts consistent with the parametric approach in general. This leaves open the possibility that the finite-state approach may dominate the functional approximation approach for some problems because of the applicability of GaussSeidel acceleration ideas.
12.8 Parametric Approximations and Simulation Methods

The main idea of parametric dynamic programming methods is to parameterize the critical functions and find some parameter choice which generates a good approximation. One direct and simple implementation of that idea is to parameterize the control law, $\hat{U}(x ; a)$, and through simulation find that coefficient choice. $a$. which generates the greatest value. In this section we will discuss a simple application of that approach.

Consider again the stochastic growth problem:

$$
V(k)=\max _c u(c)+\beta E\{V(k-c+\theta f(k-c)) \mid k, c\},
$$

where the $\theta$ are i.i.d. productivity shocks affecting the net output function $f$ : this is a special case of (12.1.20). For smooth concave problems we know that the true policy and value functions, $C(k)$ and $V(k)$, are smooth functions. increasing in $k$. In this section we will use a simple simulation approach to solve the stochastic growth problem (12.8.1).

Instead of parameterizing $C(k)$, we parameterize the savings function. $S(k) \equiv k-C(k)$. We know that $S$ is increasing but that $S\left(k_1\right)-S\left(k_2\right) \leq k_1-k_2$ for $k_1 \geq k_2$; these properties allow us to examine a simple class of rules. We will examine linear rules; hence $\hat{S}(k)=a+b k$ for coefficients $a$ and $b$ where $b \in(0,1)$. We will use simulation to approximate the value of a savings rule. Suppose that $\theta_t, t=1, \ldots, T$ is a sequence of productivity shocks. Then, for a given initial value of $k_0$, the resulting paths for $c_t$ and $k_t$ are given by $c_t=k_t-\dot{S}\left(k_t\right)$ and $k_{t+1}=\dot{S}\left(k_t\right)+\theta_t f\left(\dot{S}\left(k_t\right)\right)$, and the realized discounted utility is 
$$
W(\theta ; \hat{S})=\sum_{t=0}^T \beta^t u\left(c_t\right)
$$


We can do this for several $\theta_t$ sequences. Let $\theta^t$ be the $i$ th sequence drawn and let $c_t^{\prime}$ be the resulting consumption when we compute (12.8.2). The value of a rule $\hat{S}(k)$ beginning at $k_0$ is $V\left(k_0 ; \hat{S}\right)=E\{W(\theta ; \hat{S})\}$, which can be approximated by the sum

$$
\frac{1}{N} \sum_{j=1}^N W\left(\theta^j ; \hat{S}\right)=\frac{1}{N} \sum_{j=1}^N \sum_{t=0}^T \beta^t u\left(c_t^{\prime}\right)
$$


Note that (12.8.3) is essentially an integral over the space of $\theta$ series. A literal "simulation" approach would construct each $\theta$ by a sequence of i.i.d. draws.

This value of $W(\theta ; S)$ depends on the initial capital stock $k_0$, the particular realization of $\theta_t, t=1, \ldots, T$, and the choices for $a$ and $b$. The use of several $\theta$ realizations makes the average in (12.8.3) less sensitive to the particular realizations used.

Once we have a way to approximate $V\left(k_0 ; \hat{S}\right)$ for a linear rule $\hat{S}$ parameterized by $a$ and $b$, we optimize over $a$ and $b$ to approximate the optimal linear $\hat{S}(k)$ rule. Since $k_0$ is the initial capital stock in our definition of $V\left(k_0 ; \hat{S}\right)$, this approximation depends on $k_0$, whereas the optimal rule does not. To reduce the sensitivity of the chosen $S$ rule to $k_0$, we should choose $k_0$ to be close to the "average" capital stock.

This sounds easy but one can run into problems. Our procedure does not impose any restriction on $\hat{S}(k)$. In particular, $\hat{S}(k)$ could be negative at some $k$, or consumption $k-\hat{S}(k)$ could be negative, with either possibility causing problems because $f$ and $u$ are usually defined only for positive arguments. To deal with this problem, one should constrain the linear approximation, implying that $\hat{S}(k)= \min [\max [0, a+b k], k]$, or choose some transformation that avoids this possibility. The true rule may not be linear; in that case we could use a more flexible specification for $\hat{S}$.

The simulation approach is not efficient for smooth problems like (12.8.1) where we can exploit the continuity and concavity properties of the solution. However, more complex dynamic programming problems involving constraints, several dimensions, and/or integer variables may be approximately solved by simulation strategies. The key idea is to parameterize a family of control laws, use simulation to evaluate them, and choose the best. See Smith (1990) for an application of these ideas.

\subsection*{12.9 Shape-Preserving Methods}

The parametric approach to dynamic programming is promising but can fail if we are not careful. To illustrate this, consider the problem (12.5.1) with a very concave utility function. In figure 12.2, we display the results of a typical value function iteration. Suppose that we have chosen $x_i, i=1, \ldots, 5$, for the nodes of the approximation and that we computed $v_1, v_2, v_3, v_4$, and $v_5$ as in figure 12.2 . These five points appear to be consistent with an increasing and concave and value function. However, applying interpolation to these data to fit $\hat{V}(x ; a)$ may produce a curve, neither concave nor monotone increasing, such as $\hat{V}(x ; a)$ in figure 12.2. Even worse is that the maximum of $\hat{V}(x ; a)$ is a point between $x_2$ and $x_3$, and even exceeds the maximum $v_1$ values.

While these internodal fluctuations are consistent with the approximation theory of chapter 6 , they can wreck havoc with dynamic programming. For example, suppose that the true $V$ is increasing and concave but that at some iteration $\hat{V}$ looks like $\hat{V}$ in figure 12.2. In the next iteration, $x_M$ will be considered a very desirable state, and controls will be chosen to push the state towards $x_M$. The artificially high value of $\hat{V}\left(x_M\right)$ will lead to artificially high values for $\hat{V}\left(x_2\right)$ and $\hat{V}\left(x_3\right)$ in the next maximization step. The errors at $x_2$ and $x_3$ could interact with the values computed elsewhere to produce even worse internodal oscillations at the next fitting stage. Once this process begins, it can feed on itself and destabilize the value iteration procedure.

The problem here is the absence of shape-preservation in the algorithm. Shapepreservation is valuable property, particularly in concave problems. If $\hat{V}$ and $\pi(x, u)$ are concave in $x$ and $u$, the maximization step is a concave problem; hence the global maximum is the unique local maximum and easy to find. Furthermore $T$ is a shapepreserving operator; that is, if $\pi(x, u)$ and the conditional expectation is concave in $(x, u)$ for concave $V$, then $T V$ is concave if $V$ is concave. Therefore, if $\hat{V}$ is concave

in $x$, then the $(T \hat{V})\left(x_i\right)$ points will be concave, and a shape-preserving scheme will cause $(\hat{T} \hat{V})(x)$ to be a concave function of $x$. Approximation methods should match the shape properties of the approximated objects.

This does not mean that we can't use the polynomial methods. In some instances these problems do not arise. For example, if $V(x)$ is $C^{\infty}$ with well-behaved highorder derivatives, then orthogonal polynomial approximation is a good approximation choice, and there is less chance of these problems arising.

However, we still want to find more reliable procedures. Following is a discussion of methods that avoid these problems by design. The key idea is the use of shapepreserving approximation methods. Discretization methods will preserve shape and avoid these problems. More promising are the shape-preserving methods discussed in chapter 6.

\subsection*{Linear Interpolation}
For one-dimensional problems disruptive internodal oscillations can be avoided by the simplest of all interpolation schemes-linear interpolation. Furthermore, as discussed in chapter 6 , linear interpolation is shape-preserving. Therefore, if the $v_i$ points are increasing and concave, so will be the interpolating function.

The problem with linear interpolation is that it makes the maximization step less efficient. The kinks in a linear interpolant will generally produce kinks in the objective of the maximization step, forcing us to use slower optimization algorithms. Kinks in the value function may cause the approximate policy function to be discontinuous, an unappealing property if the true policy function is continuous. Using linear interpolation is a costly way of preserving shape.

\subsection*{Multilinear Interpolation Methods}
In multidimensional problems there are a couple of easy well-behaved approximation schemes which one can use. First, one could use the multilinear or simplicial interpolation methods discussed in chapter 6. The DYGAM package discussed in Dantzig et al. (1974) used multilinear interpolation. These methods eliminates problematic internodal oscillations since $\hat{V}$ at each point in the interior of a box is a convex combination of the values at the vertices. The fact that the interpolation scheme is monotone in the interpolation data means that $\hat{T}$ is monotone. Furthermore $\hat{T}$ inherits the contraction properties of $T$. Multilinear interpolation is costly to compute; a less costly alternative is multidimensional simplicial interpolation, as discussed in chapter 6.

Unfortunately, multilinear and simplicial interpolation have the same problems of one-dimensional linear interpolation and more. First, they preserve positivity and monotonicity but not concavity. Second, they also produce kinks in the value function and discontinuities in the policy function. These problems can be ameliorated by cutting the state space into small boxes, but only at substantial cost.

\subsection*{Schumaker Shape-Preserving Splines}
For one-dimensional dynamic programming problems with smooth concave payoff and transition functions and concave $C^1$ solutions, the Schumaker quadratic shapepreserving spline procedure will produce $C^1$ approximations of the value function and continuous policy functions. The objective in the maximization step will always be concave and $C^{\prime}$, allowing us to use a rapid scheme such as Newton's method. We can also use a small number of approximation nodes since we do not need to worry about disruptive internodal fluctuations.

Judd and Solnick (1994) apply Shumaker's quadratic splines to the single good, deterministic optimal growth problem, (12.5.1). They found that the resulting approximations were very good, substantially dominating other methods. For example, the shape-preserving method using 12 nodes did as well as linear interpolation using 120 nodes and the discrete-state approximation using 1,200 points.

Shape-preserving Hermite interpolation is particularly valuable. In this approach, after one computes the value function at a node, one also uses the envelope theorem to compute the slope of the value function at essentially zero computational cost. This slope information can be used along with the level information in the Schumaker shape preserving scheme to arrive at an even better, but still stable. approximation of value function iteration. Examples in Judd and Solnick indicate that shape-preserving Hermite interpolation will produce highly accurate solutions using few approximation nodes.

\end{document}