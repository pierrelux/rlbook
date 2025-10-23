# Midterm Question: Two Approaches to Approximate Dynamic Programming

## Setup

Consider the Bellman optimality equation:

$$
v(s) = (\mathrm{L}v)(s) = \max_{a \in \mathcal{A}} \left\{ r(s,a) + \gamma \int_{\mathcal{S}} p(s'|s,a) v(s') \, ds' \right\}.
$$

You want to find $v^*$ satisfying $v^* = \mathrm{L}v^*$, but you cannot represent arbitrary functions on a computer. Instead, you approximate using:

$$
\hat{v}(s) = \sum_{i=1}^n a_i \varphi_i(s),
$$

where $\{\varphi_1, \ldots, \varphi_n\}$ are fixed basis functions and $a = (a_1, \ldots, a_n)$ are coefficients to determine.

You choose $n$ collocation points $\{s_1, \ldots, s_n\}$ where you'll enforce conditions.

**Recall:** The **residual function** is $R(s; a) = \mathrm{L}\hat{v}(s; a) - \hat{v}(s; a)$. A true solution has $R(s; a) = 0$ everywhere.

---

## Part A: Understanding Two Different Approaches (30 points)

### Approach 1: Minimize the Residual Directly

One idea is to find coefficients $a$ that make the residual small:

$$
\min_{a} \sum_{i=1}^n \left[ \mathrm{L}\hat{v}(s_i; a) - \hat{v}(s_i; a) \right]^2.
$$

### Approach 2: Successive Approximation with Projection

Start with initial coefficients $a^{(0)}$. At each iteration $k$:

1. Compute target values: $t_i^{(k)} = (\mathrm{L}\hat{v}^{(k)})(s_i)$ for each collocation point $s_i$
2. Find new coefficients $a^{(k+1)}$ by minimizing the L2 fitting error to these targets:
   $$
   \min_{a} \sum_{i=1}^n \left[ \hat{v}(s_i; a) - t_i^{(k)} \right]^2
   $$
3. Repeat

---

**Question A.1** (12 points): Both Use L2 Loss - But Very Differently!

Notice that both approaches minimize an L2 (squared) loss. But look carefully at what's inside each squared term.

(a) In Approach 1, the objective is:
$$
\min_{a} \sum_{i=1}^n \left[ \mathrm{L}\hat{v}(s_i; a) - \hat{v}(s_i; a) \right]^2.
$$

Where does the variable $a$ (that you're optimizing over) appear in this expression? Be specific - does it appear in both terms inside the square brackets, or just one?

(b) In Approach 2, the objective at iteration $k$ is:
$$
\min_{a} \sum_{i=1}^n \left[ \hat{v}(s_i; a) - t_i^{(k)} \right]^2.
$$

Where does the variable $a$ appear in this expression? Remember that $t_i^{(k)} = \mathrm{L}\hat{v}^{(k)}(s_i)$ was already computed in step 1 using the OLD coefficients $a^{(k)}$.

(c) This difference has major computational implications. In Approach 1, to compute the gradient $\frac{\partial}{\partial a_j}[\text{objective}]$, you need to differentiate through the term $\mathrm{L}\hat{v}(s_i; a)$ with respect to $a$. What makes this difficult? (Hint: Look at the definition of $\mathrm{L}$ - what operation does it involve?)

(d) In Approach 2, to compute the gradient $\frac{\partial}{\partial a_j}[\text{objective}]$ at iteration $k$, do you need to differentiate through $\mathrm{L}$? Why or why not? What kind of optimization problem is step 2 of Approach 2? (Hint: The targets $t_i^{(k)}$ are just fixed numbers at iteration $k$.)

---

**Question A.2** (10 points): What Problem Are You Solving?

(a) In Approach 1, you're searching for coefficients $a^*$ such that the residual $R(\cdot; a^*)$ is minimized. Once you find $a^*$, you're done. 

In Approach 2, at iteration $k$ you have coefficients $a^{(k)}$. After step 2, you've made the residual $R^{(k)}(s_i) = \mathrm{L}\hat{v}^{(k)}(s_i) - \hat{v}^{(k)}(s_i)$ equal to zero at all collocation points (verify this for yourself). 

But then you immediately compute NEW target values $t_i^{(k+1)} = \mathrm{L}\hat{v}^{(k+1)}(s_i)$ in the next iteration. 

**Question:** Does this mean $t_i^{(k+1)} = t_i^{(k)}$? That is, are the target values the same across iterations? Explain your reasoning.

(b) Based on your answer to part (a), is Approach 2 searching for a single set of coefficients $a^*$ that minimizes the residual, or is it doing something else? Explain in 3-4 sentences what Approach 2 is actually doing.

---

**Question A.3** (10 points): The Conceptual Difference

Here's a key observation: In Approach 2, when we compute $f^{(k)}(s) = \mathrm{L}\hat{v}^{(k)}(s)$, this function $f^{(k)}$ is typically **not** in the span of our basis functions $\{\varphi_1, \ldots, \varphi_n\}$.

(a) Explain in your own words why $f^{(k)} = \mathrm{L}\hat{v}^{(k)}$ is generally not in $\text{span}\{\varphi_1, \ldots, \varphi_n\}$ even though $\hat{v}^{(k)}$ is. (Hint: Think about what the operator $\mathrm{L}$ does - does it preserve the form of the function?)

(b) Given that $f^{(k)}$ is not in our approximation space, what are we doing in step 2 of Approach 2 when we solve $\hat{v}^{(k+1)}(s_i) = t_i^{(k)}$? (Hint: We're approximating $f^{(k)}$ with something in our basis.)

(c) This is why Approach 2 is called "successive approximation with projection." Explain in 2-3 sentences:
   - What does "successive approximation" refer to?
   - What does "projection" refer to?
   - How does this differ from Approach 1's goal of finding $a^*$ to minimize the residual?

---

## Part B: Synthesis (15 points)

Write a paragraph (6-8 sentences) that synthesizes the key conceptual difference between these approaches. Address the following points:

1. What is the fundamental difference in what each approach is trying to solve?
2. Why does Approach 1 require differentiating through the Bellman operator $\mathrm{L}$ while Approach 2 does not?
3. Which approach aligns better with the fixed-point nature of the Bellman equation ($v = \mathrm{L}v$)? Explain your reasoning.
4. If the Bellman operator is a contraction (as it is for discounted MDPs), which approach do you expect to have better convergence properties and why?

Your answer should demonstrate conceptual understanding, not just restate the procedures.

---

## Grading Rubric

**Part A (32 points):**
- A.1: Where a appears in Approach 1 (2), where a appears in Approach 2 (2), difficulty of Approach 1 gradient (3), Approach 2 is standard supervised learning (3)
- A.2: Target values across iterations (5), what Approach 2 is doing (5)
- A.3: Why L maps outside span (3), what step 2 does (3), synthesis of "successive approximation with projection" (4)

**Part B (15 points):**
- Fundamental difference (4)
- Why gradient requirements differ (3)
- Fixed-point alignment (4)
- Convergence discussion (4)

**Total: 47 points**

---

## Instructor Notes

**Pedagogical choice: Making both L2 losses explicit**

The question deliberately shows that BOTH approaches use L2 loss, but in fundamentally different ways:
- **Approach 1**: $\min_a \sum [L\hat{v}(s_i; a) - \hat{v}(s_i; a)]^2$ — variable $a$ appears inside $L$, need to differentiate through max
- **Approach 2**: $\min_a \sum [\hat{v}(s_i; a) - t_i^{(k)}]^2$ — targets $t_i^{(k)}$ are fixed, standard supervised learning

This helps students see that the issue isn't "optimization vs. something else" — both optimize. The key difference is WHERE the parameters appear and WHAT you're fitting to.

**Core concepts tested:**
1. Understanding that applying $\mathrm{L}$ maps outside the approximation space
2. Recognizing the difference between "minimize residual" vs. "iterate apply-and-project"  
3. Appreciating why successive approximation avoids differentiation through max
4. Understanding the role of contraction properties
5. Seeing that both approaches use L2 loss but in fundamentally different ways

**Key insights expected:**

*Question A.1(a):* In Approach 1, $a$ appears in BOTH terms: $\mathrm{L}\hat{v}(s_i; a)$ and $\hat{v}(s_i; a)$. The coefficients are inside the operator.

*Question A.1(b):* In Approach 2, $a$ appears ONLY in $\hat{v}(s_i; a)$. The targets $t_i^{(k)}$ are fixed numbers (already computed).

*Question A.1(c):* Differentiating $\mathrm{L}\hat{v}$ w.r.t. $a$ requires the chain rule through the max operator, which is non-differentiable (or requires subdifferential calculus).

*Question A.1(d):* No! Approach 2's step 2 is just linear least squares (or ordinary supervised learning). The gradient is standard: $\frac{\partial}{\partial a_j} \sum [\hat{v}(s_i; a) - t_i^{(k)}]^2$ only involves the basis functions, not the operator $\mathrm{L}$.

*Question A.2(a):* No, $t_i^{(k+1)} \neq t_i^{(k)}$ because $\hat{v}^{(k+1)} \neq \hat{v}^{(k)}$, so applying $\mathrm{L}$ gives different target values.

*Question A.2(b):* Approach 2 is NOT minimizing the residual of fixed coefficients. It's iterating: apply $\mathrm{L}$ to current approximation, get targets, fit to those targets (standard supervised learning), repeat. Each iteration solves a different least-squares problem.

*Question A.3(a):* The operator $\mathrm{L}$ involves max, expectation, and composition with the approximation $\hat{v}$. Even if $\hat{v}$ is a linear combination of basis functions, $\mathrm{L}\hat{v}$ will generally be a more complicated function (e.g., max of linear functions is piecewise linear, not linear).

*Question A.3(c):* 
- "Successive approximation" = iterate the map $v \mapsto \mathrm{L}v$
- "Projection" = force the result back into $\text{span}\{\varphi_i\}$ by collocation
- Approach 1 seeks one optimal $a^*$; Approach 2 iterates in function space

**Part B expectations:**
Students should articulate that Approach 1 treats this as an optimization problem in coefficient space, while Approach 2 treats it as iteration in function space with projection back to the approximation space. The gradient issue arises in Approach 1 because we're optimizing over coefficients that appear inside $\mathrm{L}$. Approach 2 aligns better with $v = \mathrm{L}v$ as a fixed-point equation. The contraction property guarantees Approach 2 converges but doesn't make Approach 1's objective convex.

**What makes this accessible yet rigorous:**
- Both approaches use L2 loss (familiar to students) but in different ways
- Questions ask "where does $a$ appear?" rather than "derive the gradient"
- More "explain in words" questions
- Focus on conceptual understanding over manipulation
- Concrete guidance about what to think about
- Students can recognize Approach 2 as "standard supervised learning with changing targets"

**Connection to practice:**
After grading, mention that most successful deep RL (DQN, etc.) uses Approach 2-style methods. Approach 1-style residual minimization has theoretical appeal but practical challenges.
