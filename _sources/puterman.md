# 4.3 OPTIMALITY EQUATIONS AND THE PRINCIPLE OF OPTIMALITY

In this section we introduce optimality equations (sometimes referred to as Bellman equations or functional equations) and investigate their properties. We show that solutions of these equations correspond to optimal value functions and that they also provide a basis for determining optimal policies. We assume either finite or countable $S$ to avoid technical subtleties.

Let

$$u_t^*(h_t) = \sup_{\pi \in \Pi^{\text{HR}}} u_t^{\pi}(h_t). \tag{4.3.1}$$

It denotes the supremum over all policies of the expected total reward from decision epoch $t$ onward when the history up to time $t$ is $h_t$. For $t > 1$, we need not consider all policies when taking the above supremum. Since we know $h_t$, we only consider portions of policies from decision epoch $t$ onward; that is, we require only the supremum over $(d_t, d_{t+1}, \ldots, d_{N-1}) \in D_t^{\text{HR}} \times D_{t+1}^{\text{HR}} \times \cdots \times D_{N-1}^{\text{HR}}$. When minimizing costs instead of maximizing rewards, we sometimes refer to $u_t^*$ as a *cost-to-go* function.

The *optimality equations* are given by

$$u_t(h_t) = \sup_{a \in A_{s_t}} \left\{ r_t(s_t, a) + \sum_{j \in S} p_t(j|s_t, a) u_{t+1}(h_t, a, j) \right\} \tag{4.3.2}$$

for $t = 1, \ldots, N - 1$ and $h_t = (h_{t-1}, a_{t-1}, s_t) \in H_t$. For $t = N$, we add the boundary condition

$$u_N(h_N) = r_N(s_N) \tag{4.3.3}$$

for $h_N = (h_{N-1}, a_{N-1}, s_N) \in H_N$.

These equations reduce to the policy evaluation equations (4.2.1) when we replace the supremum over all actions in state $s_t$ by the action corresponding to a specified policy, or equivalently, when $A_s$ is a singleton for each $s \in S$.

The operation "sup" in (4.3.2) is implemented by evaluating the quantity in brackets for each $a \in A_{s_t}$ and then choosing the supremum over all of these values. When $A_{s_t}$ is a continuum, the supremum might be found analytically. If the supremum in (4.3.2) is attained, for example, when each $A_{s_t}$ is finite, it can be replaced by "max" so (4.3.2) becomes

$$u_t(h_t) = \max_{a \in A_{s_t}} \left\{ r_t(s_t, a) + \sum_{j \in S} p_t(j|s_t, a) u_{t+1}(h_t, a, j) \right\}. \tag{4.3.4}$$

A solution to the system of equations (4.3.2) or (4.3.4) and boundary condition (4.3.3) is a sequence of functions $u_t: H_t \to R$, $t = 1, \ldots, N$, with the property that $u_N$ satisfies (4.3.3), $u_{N-1}$ satisfies the $(N - 1)$th equation with $u_N$ substituted into the right-hand side of the $(N - 1)$th equation, and so forth.

# 84 FINITE-HORIZON MARKOV DECISION PROCESSES

The optimality equations are fundamental tools in Markov decision theory, having the following important and useful properties.

**a.** Solutions to the optimality equations are the optimal returns from period $t$ onward for each $t$.

**b.** They provide a method for determining whether a policy is optimal. If the expected total reward of policy $\pi$ from period $t$ onward satisfies this system of equations for $t = 1, \ldots, N$ it is optimal.

**c.** They are the basis for an efficient procedure for computing optimal return functions and policies.

**d.** They may be used to determine structural properties of optimal policies and return functions.

Before stating and proving the main result in this chapter, we introduce the following important yet simple lemma.

**Lemma 4.3.1.** Let $w$ be a real-valued function on an arbitrary discrete set $W$ and let $q(\cdot)$ be a probability distribution on $W$. Then

$$\sup_{u \in W} w(u) \geq \sum_{u \in W} q(u)w(u)$$

*Proof.* Let $w^* = \sup_{u \in W} w(u)$. Then,

$$w^* = \sum_{u \in W} q(u)w^* \geq \sum_{u \in W} q(u)w(u).$$

$\square$

Note that the lemma remains valid with $W$ a Borel subset of a measurable space, $w(u)$ an integrable function on $W$, and the summation replaced by integration.

The following theorem summarizes the optimality properties of solutions of the optimality equation. Its inductive proof illustrates several dynamic programming principles and consists of two parts. First we establish that solutions provide upper bounds on $u_t^*$ and then we establish existence of a policy $\pi'$ for which $u_t^{\pi'}$ is arbitrarily close to $u_t$.

**Theorem 4.3.2.** Suppose $u_t$ is a solution of (4.3.2) for $t = 1, \ldots, N - 1$, and $u_N$ satisfies (4.3.3). Then

**a.** $u_t(h_t) = u_t^*(h_t)$ for all $h_t \in H_t$, $t = 1, \ldots, N$, and

**b.** $u_1(s_1) = v_N^*(s_1)$ for all $s_1 \in S$.

*Proof.* The proof is in two parts. First we establish by induction that $u_n(h_n) \geq u_n^*(h_n)$ for all $h_n \in H_n$ and $n = 1, 2, \ldots, N$.

Since no decision is made in period $N$, $u_N(h_N) = r_N(s_N) = u_N^*(h_N)$ for all $h_N \in H_N$ and $\pi \in \Pi^{\text{HR}}$. Therefore $u_N(h_N) = u_N^*(h_N)$ for all $h_N \in H_N$. Now assume

# OPTIMALITY EQUATIONS AND THE PRINCIPLE OF OPTIMALITY 85

that $u_t(h_t) \geq u_t^*(h_t)$ for all $h_t \in H_t$ for $t = n + 1, \ldots, N$. Let $\pi' = (d_1', d_2', \ldots, d_{N-1}')$ be an arbitrary policy in $\Pi^{\text{HR}}$. For $t = n$, the optimality equation is

$$u_n(h_n) = \sup_{a \in A_{s_n}} \left\{ r_n(s_n, a) + \sum_{j \in S} p_n(j|s_n, a) u_{n+1}(h_n, a, j) \right\}.$$

By the induction hypothesis

$$u_n(h_n) \geq \sup_{a \in A_{s_n}} \left\{ r_n(s_n, a) + \sum_{j \in S} p_n(j|s_n, a) u_{n+1}^*(h_n, a, j) \right\} \tag{4.3.5}$$

$$\geq \sup_{a \in A_{s_n}} \left\{ r_n(s_n, a) + \sum_{j \in S} p_n(j|s_n, a) u_n^{\pi'}(h_n, a, j) \right\} \tag{4.3.6}$$

$$\geq \sum q_{d_n'(h_n)}(a) \left\{ r_n(s_n, a) + \sum_{j \in S} p_n(j|s_n, a) u_{n+1}^{\pi'}(h_n, a, j) \right\} \tag{4.3.7}$$

$$= u_n^{\pi'}(h_n).$$

The inequality in (4.3.5) follows from the induction hypothesis and the non-negativity of $p_n$, and that in (4.3.6) from the definition of $u_{n+1}^*$. That in (4.3.7) follows from Lemma 4.3.1 with $W = A_{s_n}$ and $w$ equal to the expression in brackets. The last equality follows from (4.2.6) and Theorem 4.2.2. Since $\pi'$ is arbitrary,

$$u_n(h_n) \geq u_n^{\pi}(h_n) \quad \text{for all } \pi \in \Pi^{\text{HR}}.$$

Thus $u_n(h_n) \geq u_n^*(h_n)$ and the induction hypothesis holds.

Now we establish that for any $\varepsilon > 0$, there exists a $\pi' \in \Pi^{\text{HD}}$ for which

$$u_n^{\pi'}(h_n) + (N - n)\varepsilon \geq u_n(h_n) \tag{4.3.8}$$

for all $h_n \in H_n$ and $n = 1, 2, \ldots, N$. To do this, construct a policy $\pi' = (d_1, d_2, \ldots, d_{N-1})$ by choosing $d_n(h_n)$ to satisfy

$$r_n(s_n, d_n(h_n)) + \sum_{j \in S} p_n(j|s_n, d_n(h_n)) u_{n+1}(s_n, d_n(h_n), j) + \varepsilon \geq u_n(h_n).$$

$$(4.3.9)$$

We establish (4.3.8) by induction. Since $u_N^{\pi}(h_N) = u_N(h_N)$, the induction hypothesis holds for $t = N$. Assume that $u_t^{\pi}(h_t) + (N - t)\varepsilon \geq u_t(h_t)$ for $t = n + 1, \ldots, N$.

# 86 FINITE-HORIZON MARKOV DECISION PROCESSES

Then it follows from Theorem 4.2.1 and (4.3.9) that

$$u_n^{\pi'}(h_n) = r_n(s_n, d_n(h_n)) + \sum_{j \in S} p_n(j|s_n, d_n(h_n)) u_{n+1}^{\pi'}(s_n, d_n(h_n), j)$$

$$\geq r_n(s_n, d_n(h_n)) + \sum_{j \in S} p_n(j|s_n, d_n(h_n)) u_{n+1}(s_n, d_n(h_n), j)$$

$$-(N - n - 1)\varepsilon$$

$$\geq u_n(h_n) - (N - n)\varepsilon.$$

Thus the induction hypothesis is satisfied and (4.3.8) holds for $n = 1, 2, \ldots, N$.

Therefore for any $\varepsilon > 0$, there exists a $\pi' \in \Pi^{\text{HR}}$ for which

$$u_n^*(h_n) + (N - n)\varepsilon \geq u_n^{\pi'}(h_n) + (N - n)\varepsilon \geq u_n(h_n) \geq u_n^*(h_n)$$

so that (a) follows. Part (b) follows from the definitions of the quantities. $\square$

Part (a) of Theorem 4.3.2 means that solutions of the optimality equation are the optimal value functions from period $t$ onward, and result (b) means that the solution to the equation with $n = 1$ is the value function for the MDP, that is, it is the optimal value from decision epoch 1 onward.

The following result shows how to use the optimality equations to find optimal policies, and to verify that a policy is optimal. Theorem 4.3.3 uses optimality equations (4.3.4) in which maxima are attained. Theorem 4.3.4 considers the case of suprema.

**Theorem 4.3.3.** Suppose $u_t^*$, $t = 1, \ldots, N$ are solutions of the optimality equations (4.3.4) subject to boundary condition (4.3.3), and that policy $\pi^* = (d_1^*, d_2^*, \ldots, d_{N-1}^*) \in \Pi^{\text{HD}}$ satisfies

$$r_t(s_t, d_t^*(h_t)) + \sum_{j \in S} p_t(j|s_t, d_t^*(h_t)) u_{t+1}^*(h_t, d_t^*(h_t), j)$$

$$= \max_{a \in A_{s_t}} \left\{ r_t(s_t, a) + \sum_{j \in S} p_t(j|s_t, a) u_{t+1}^*(h_t, a, j) \right\} \tag{4.3.10}$$

for $t = 1, \ldots, N - 1$.

Then

**a.** For each $t = 1, 2, \ldots, N$,

$$u_t^{\pi^*}(h_t) = u_t^*(h_t), \quad h_t \in H_t. \tag{4.3.11}$$

**b.** $\pi^*$ is an optimal policy, and

$$v_N^{\pi^*}(s) = v_N^*(s), \quad s \in S. \tag{4.3.12}$$

# OPTIMALITY EQUATIONS AND THE PRINCIPLE OF OPTIMALITY 87

*Proof.* We establish part (a); part (b) follows from Theorem 4.2.1 and Theorem 4.3.2b. The proof is by induction. Clearly

$$u_N^{\pi^*}(h_n) = u_N^*(h_n), \quad h_n \in H_n.$$

Assume the result holds for $t = n + 1, \ldots, N$. Then, for $h_n = (h_{n-1}, d_{n-1}^*(h_{n-1})), s_n)$,

$$u_n^*(h_n) = \max_{a \in A_{s_n}} \left\{ r_n(s_n, a) + \sum_{j \in S} p_n(j|s_n, a) u_{n+1}^*(h_n, a, j) \right\}$$

$$= r_n(s_n, d_n^*(h_n)) + \sum_{j \in S} p_n(j|s_n, d_n^*(h_n)) u_{n+1}^*(h_n, d_n^*(h_n), j)$$

$$= u_n^{\pi^*}(h_n).$$

The second equality is a consequence of (4.3.10), and the induction hypothesis and the last equality follows from Theorem 4.2.1. Thus the induction hypothesis is satisfied and the result follows. $\square$

We frequently write equation (4.3.10) as

$$d_t^*(h_t) \in \arg\max_{a \in A_{s_t}} \left\{ r_t(s_t, a) + \sum_{j \in S_t} p_t(j|s_t, a) u_{t+1}^*(h_t, a, j) \right\},$$

where "arg max" has been defined in Sec. 2.2.

The theorem implies that an optimal policy is found by first solving the optimality equations, and then for each history choosing a decision rule which selects any action which attains the maximum on the right-hand side of (4.3.10) for $t = 1, 2, \ldots, N$. When using this equation in computation, for each history the right-hand side is evaluated for all $a \in A_{s_t}$ and the set of maximizing actions is recorded. When there is more than one maximizing action in this set, there is more than one optimal policy.

Note that we have restricted attention to history-dependent deterministic policies in Theorem 4.3.3. This is because if there existed a history-dependent randomized policy which satisfied the obvious generalization of (4.3.10), as a result of Lemma 4.3.1, we could find a deterministic policy which satisfied (4.3.10). We expand on this point in the next section.

This theorem provides a formal statement of "The Principle of Optimality," a fundamental result of dynamic programming. An early verbal statement appeared in Bellman (1957, p. 83).

"An optimal policy has the property that whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision."

Denardo (1982, p. 15) provides a related statement.

"There exists a policy that is optimal for every state (at every stage)."

# 88 FINITE-HORIZON MARKOV DECISION PROCESSES

Any deterministic policy $\pi^*$ satisfying (4.3.10) has these properties and such a policy must *exist* because maxima are attained. Alternatively, Theorem 4.3.3 provides *sufficient* conditions for verifying the optimality of a policy.

Note that "The Principle of Optimality" may not be valid for other optimality criteria. In Sec. 4.6.2, we provide an example in which it does not hold, albeit under a nonstandard optimality criteria.

In case the supremum in (4.3.2) is not attained, the decision maker must be content with $\varepsilon$-optimal policies. To account for this we modify Theorem 4.3.3 as follows. Arguments in the second part of the proof of Theorem 4.3.2 can be used to establish it.

**Theorem 4.3.4.** Let $\varepsilon > 0$ be arbitrary and suppose $u_t^*$, $t = 1, \ldots, N$ are solutions of the optimality equations (4.3.2) and (4.3.3). Let $\pi^{\varepsilon} = (d_1^{\varepsilon}, d_2^{\varepsilon}, \ldots, d_{N-1}^{\varepsilon}) \in \Pi^{\text{HD}}$ satisfy

$$r_t(s_t, d_t^{\varepsilon}(h_t)) + \sum_{j \in S} p_t(j|s_t, d_t^{\varepsilon}(h_t)) u_{t+1}^*(h_t, d_t^{\varepsilon}(h_t), j) + \frac{\varepsilon}{N-1}$$

$$\geq \sup_{a \in A_{s_t}} \left\{ r_t(s_t, a) + \sum_{j \in S} p_t(j|s_t, a) u_{t+1}^*(h_t, a, j) \right\} \tag{4.3.13}$$

for $t = 1, 2, \ldots, N - 1$. Then,

**a.** For each $t = 1, 2, \ldots, N - 1$,

$$u_t^{\pi^{\varepsilon}}(h_t) + (N - t)\frac{\varepsilon}{N-1} \geq u_t^*(h_t), \quad h_t \in H_t. \tag{4.3.14}$$

**b.** $\pi^{\varepsilon}$ is an $\varepsilon$-optimal policy, that is

$$v_N^{\pi^{\varepsilon}}(s) + \varepsilon \geq v_N^*(s), \quad s \in S. \tag{4.3.15}$$

## 4.4 OPTIMALITY OF DETERMINISTIC MARKOV POLICIES

This section provides conditions under which there exists an optimal policy which is deterministic and Markovian, and illustrates how backward induction can be used to determine the structure of an optimal policy.

From the perspective of application, we find it comforting that by restricting attention to nonrandomized Markov policies, which are simple to implement and evaluate, we may achieve as large an expected total reward as if we used randomized history-dependent policies. We show that when the immediate rewards and transition probabilities depend on the past only through the current state of the system (as assumed throughout this book), the optimal value functions depend on the history only through the current state of the system. This enables us to impose assumptions on the action sets, rewards, and transition probabilities which ensure existence of optimal policies which depend only on the system state.

# OPTIMALITY OF DETERMINISTIC MARKOV POLICIES 89

Inspection of the proof of Theorem 4.3.2 reveals that it constructs an $\varepsilon$-optimal deterministic history-dependent policy. Theorem 4.3.3 and 4.3.4 identify optimal and $\varepsilon$-optimal policies. We summarize these results as follows.

## Theorem 4.4.1.

**a.** For any $\varepsilon > 0$, there exists an $\varepsilon$-optimal policy which is deterministic history dependent. Any policy in $\Pi^{\text{HD}}$ which satisfies (4.3.13) is $\varepsilon$-optimal.

**b.** Let $u_t^*$ be a solution of (4.3.2) and (4.3.3) and suppose that for each $t$ and $s_t \in S$, there exists an $a' \in A_{s_t}$ for which

$$r_t(s_t, a') + \sum_{j \in S} p_t(j|s_t, a') u_{t+1}^*(h_t, a', j)$$

$$= \sup_{a \in A_{s_t}} \left\{ r_t(s_t, a) + \sum_{j \in S} p_t(j|s_t, a) u_{t+1}^*(h_t, a, j) \right\} \tag{4.4.1}$$

for all $h_t = (s_{t-1}, a_{t-1}, s_t) \in H_t$. Then there exists a deterministic history-dependent policy which is optimal.

*Proof.* Part (a) follows from the second part of the proof of Theorem 4.3.2. The policy $\pi^{\varepsilon}$ in Theorem 4.3.4 is optimal and deterministic. When there exists an $a' \in A_{s_t}$ for which (4.4.1) holds, the policy $\pi^* \in \Pi^{\text{HD}}$ of Theorem 4.3.3 is optimal. $\square$

We next show by induction that there exists an optimal policy which is Markovian and deterministic.

**Theorem 4.4.2.** Let $u_t^*$, $t = 1, \ldots, N$ be solutions of (4.3.2) and (4.3.3). Then

**a.** For each $t = 1, \ldots, N$, $u_t^*(h_t)$ depends on $h_t$ only through $s_t$.

**b.** For any $\varepsilon > 0$, there exists an $\varepsilon$-optimal policy which is deterministic and Markov.

**c.** If there exists an $a' \in A_{s_t}$ such that (4.4.1) holds for each $s_t \in S$ and $t = 1, 2, \ldots, N - 1$, there exists an optimal policy which is deterministic and Markov.

*Proof.* We show that (a) holds by induction. Since $u_N^*(h_N) = u_N^*(h_{N-1}, a_{N-1}, s) = r_N(s)$ for all $h_{N-1} \in H_{N-1}$ and $a_{N-1} \in A_{s_{N-1}}$, $u_N^*(h_N) = u_N^*(s_N)$. Assume now that (a) is valid for $n = t + 1, \ldots, N$. Then

$$u_t^*(h_t) = \sup_{a \in A_{s_t}} \left\{ r_t(s_t, a) + \sum_{j \in S} p_t(j|s_t, a) u_{t+1}^*(h_t, a, j) \right\},$$

which by the induction hypothesis gives

$$u_t^*(h_t) = \sup_{a \in A_{s_t}} \left\{ r_t(s_t, a) + \sum_{j \in S} p_t(j|s_t, a) u_{t+1}^*(j) \right\}. \tag{4.4.2}$$

Since the quantity in brackets depend on $h_t$ only through $s_t$, (a) holds for all $t$.

# 90 FINITE-HORIZON MARKOV DECISION PROCESSES

Choose $\varepsilon > 0$, and let $\pi^{\varepsilon} = (d_1^{\varepsilon}, d_2^{\varepsilon}, \ldots, d_{N-1}^{\varepsilon})$ be any policy in $\Pi^{\text{MD}}$ satisfying

$$r_t(s_t, d_t^{\varepsilon}(s_t)) + \sum_{j \in S} p_t(j|s_t, d_t^{\varepsilon}(s_t)) u_{t+1}^*(j) + \frac{\varepsilon}{N-1}$$

$$\geq \sup_{a \in A_{s_t}} \left\{ r_t(s_t, a) + \sum_{j \in S} p_t(j|s_t, a) u_{t+1}^*(j) \right\}. \tag{4.4.3}$$

Then, by part (a), $\pi^{\varepsilon}$ satisfies the hypotheses of Theorem 4.3.4b so it is $\varepsilon$-optimal.

Part (c) follows by noting that under the hypotheses of Theorem 4.4.1b, there exists a $\pi^* = (d_1^*, d_2^*, \ldots, d_{N-1}^*) \in \Pi^{\text{MD}}$, which satisfies

$$r_t(s_t, d_t^*(s_t)) + \sum_{j \in S} p_t(j|s_t, d_t^*(s_t)) u_{t+1}^*(j)$$

$$= \max_{a \in A_{s_t}} \left\{ r_t(s_t, a) + \sum_{j \in S} p_t(j|s_t, a) u_{t+1}^*(j) \right\}. \tag{4.4.4}$$

Therefore by part (a) and Theorem 4.3.3b, $\pi^*$ is optimal. $\square$

Thus we have established that

$$v_N^*(s) = \sup_{\pi \in \Pi^{\text{HR}}} v_N^{\pi}(s) = \sup_{\pi \in \Pi^{\text{MD}}} v_N^{\pi}(s), \quad s \in S.$$

We now provide conditions under which the supremum in (4.4.1) is attained, so that we may easily determine when there exists a deterministic Markovian policy which is optimal. Following Appendix B, we say that a set $X$ is *compact* if it is a compact subset of a complete separable metric space. In many applications we consider only compact subsets of $R^n$. The proof of part (c) below is quite technical and relies on properties of semicontinuous functions described in Appendix B. Note that more subtle arguments are required for general $S$.

**Proposition 4.4.3.** Assume $S$ is finite or countable, and that

**a.** $A_s$ is finite for each $s \in S$, or

**b.** $A_s$ is compact, $r_t(s, a)$ is continuous in $a$ for each $s \in S$, there exists an $M < \infty$ for which $|r_t(s, a)| \leq M$ for all $a \in A_s$, $s \in S$, and $p_t(j|s, a)$ is continuous in $a$ for each $j \in S$ and $s \in S$ and $t = 1, 2, \ldots, N$, or

**c.** $A_s$ is a compact, $r_t(s, a)$ is upper semicontinuous (u.s.c.) in $a$ for each $s \in S$, there exists an $M < \infty$ for which $|r_t(s, a)| \leq M$ for all $a \in A_s$, $s \in S$, and for each $j \in S$ and $s \in S$, $p_t(j|s, a)$ is lower semi-continuous (l.s.c.) in $a$ and $t = 1, 2, \ldots, N$.

Then there exists a deterministic Markovian policy which is optimal.

*Proof.* We show that there exists an $a'$ which satisfies (4.4.1) under hypothesis (a), (b), or (c), in which case the result follows from Theorem 4.4.2. Note that as a consequence of Theorem 4.4.2c, we require that, for each $s \in S$, there exists an

# OPTIMALITY OF DETERMINISTIC MARKOV POLICIES 91

$a' \in A_s$, for which

$$r_t(s, a') + \sum_{j \in S} p_t(j|s_t, a') u_{t+1}^*(j)$$

$$= \sup_{a \in A_s} \left\{ r_t(s, a) + \sum_{j \in S} p_t(j|s, a) u_{t+1}^*(j) \right\}. \tag{4.4.5}$$

Clearly such an $a'$ exists when $A_s$ is finite so the result follows under a.

Suppose that the hypotheses in part (c) hold. By assumption $|r_t(s, a)| \leq M$ for all $s \in S$ and $a \in A_s$, $|u_t^*(s)| \leq NM$ for all $s \in S$ and $t = 1, 2, \ldots, N$. Therefore, for each $t$, $u_t^*(s) - NM \leq 0$. Now apply Proposition B.3, with $s$ fixed and $X$ identified with $S_Y$, $A_s$ identified with $q(x, y)$, $p_t(j|s, a)$ identified with $f(w, x)$ and $u_t^*(s) - NM$ identified with $f(x)$, to obtain that

$$\sum_{j \in S} p_t(j|s, a)[u_{t+1}^*(j) - NM]$$

is u.s.c., from which we conclude that $\sum_{j \in S} p_t(j|s, a)u_{t+1}^*(j)$ is u.s.c. By Proposition B.1.a, $r_t(s, a) + \sum_{j \in S} p_t(j|s, a)u_{t+1}^*(j)$ is u.s.c. in $a$ for each $s \in S$. Therefore by Theorem B.2, the supremum over $a$ in (4.4.5) is attained, from which the result follows.

Conclusion (b) follows from (c) since continuous functions are both upper and lower semicontinuous. $\square$

To illustrate this result, consider the following example which we analyze in further detail in Sec. 6.4.

**Example 4.4.1.** Let $N = 3$; $S = \{s_1, s_2\}$; $A_{s_1} = [0, 2]$, and $A_{s_2} = \{a_{2,1}\}$; $r_t(s_1, a) = -a^2$, $r_t(s_2, a_{2,1}) = -\frac{1}{2}$, $p_t(s_1|s_1, a) = \frac{1}{2}a$, $p_t(s_2|s_1, a) = 1 - \frac{1}{2}a$, and $p_t(s_2|s_2, a_{2,1}) = 1$ for $t = 1, 2$, $r_3(s_1) = -1$, and $r_3(s_2) = -\frac{1}{2}$. (See Fig. 4.4.1)

Since $A_{s_1}$ is compact and $r_t(s_1, \cdot)$ and $p_t(j|s_1, \cdot)$ are continuous functions on $A_{s_1}$ there exists a deterministic Markov policy which is optimal.

[THIS IS FIGURE: A graphical representation showing two states $S_1$ and $S_2$ with arrows indicating transitions. From $S_1$, there's a self-loop labeled "{ -a², a/2 }" and an arrow to $S_2$ labeled "{ -a², 1 - a/2 }". From $S_2$, there's a self-loop labeled "{ -1/2, 1 }" and an arrow labeled "$a_{2,1}$".]

**Figure 4.4.1** Graphical representation of Example 4.4.1.

# 92 FINITE-HORIZON MARKOV DECISION PROCESSES

## 4.5 BACKWARD INDUCTION

Backward induction provides an efficient method for solving finite-horizon discrete-time MDPs. For stochastic problems, enumeration and evaluation of *all* policies is the only alternative, but forward induction and reaching methods provide alternative solution methods for deterministic systems. The terms "backward induction" and "dynamic programming" are synonymous, although the expression "dynamic programming" often refers to all results and methods for sequential decision processes. This section presents the backward induction algorithm and shows how to use it to find optimal policies and value functions. The algorithm generalizes the policy evaluation algorithm of Sec. 4.2.

We present the algorithm for a model in which maxima are obtained in (4.3.2), so that we are assured of obtaining an optimal (instead of an $\varepsilon$-optimal) Markovian deterministic policy. The algorithm solves optimality equations (4.3.4) subject to boundary condition (4.3.3). Generalization to models based on (4.3.2) is left as an exercise.

### The Backward Induction Algorithm

**1.** Set $t = N$ and

$$u_N^*(s_N) = r_N(s_N) \quad \text{for all } s_N \in S,$$

**2.** Substitute $t - 1$ for $t$ and compute $u_t^*(s_t)$ for each $s_t \in S$ by

$$u_t^*(s_t) = \max_{a \in A_{s_t}} \left\{ r_t(s_t, a) + \sum_{j \in S} p_t(j|s_t, a) u_{t+1}^*(j) \right\}. \tag{4.5.1}$$

Set

$$A_{s_t, t}^* = \arg\max_{a \in A_{s_t}} \left\{ r_t(s_t, a) + \sum_{j \in S} p_t(j|s_t, a) u_{t+1}^*(j) \right\}. \tag{4.5.2}$$

**3.** If $t = 1$, stop. Otherwise return to step 2.

As a consequence of Theorem 4.4.2, which among other results shows that $u_t^*$ depends on $h_t = (s_{t-1}, a_{t-1}, s_t)$ only through $s_t$, we need not evaluate $u_t^*$ for all $h_t \in H_t$. This significantly reduces computational effort. Combining results from Theorems 4.3.2 and 4.4.2 yields the following properties of the iterates of the backward induction algorithm.

**Theorem 4.5.1.** Suppose $u_t^*$, $t = 1, \ldots, N$ and $A_{s_t, t}^*$, $t = 1, \ldots, N - 1$ satisfy (4.5.1) and (4.5.2); then,

**a.** for $t = 1, \ldots, N$ and $h_t = (h_{t-1}, a_{t-1}, s_t)$

$$u_t^*(s_t) = \sup_{\pi \in \Pi^{\text{HR}}} u_t^{\pi}(h_t), \quad s_t \in S.$$

# BACKWARD INDUCTION 93

**b.** Let $d^*(s_t) \in A_{s_t, t}^*$ for all $s_t \in S$, $t = 1, \ldots, N - 1$, and let $\pi^* = (d_1^*, \ldots, d_{N-1}^*)$. Then $\pi^* \in \Pi^{\text{MD}}$ is optimal and satisfies

$$v_N^{\pi^*}(s) = \sup_{\pi \in \Pi^{\text{HR}}} v_N^{\pi}(s), \quad s \in S$$

and

$$u_t^{\pi^*}(s_t) = u_t^*(s_t), \quad s_t \in S$$

for $t = 1, \ldots, N$.

This theorem represents a formal statement of the following properties of the backward induction algorithm.

**a.** For $t = 1, 2, \ldots, N - 1$, it finds sets $A_{s_t, t}^*$ which contain all actions in $A_{s_t}$ which attain the maximum in (4.5.1).

**b.** It evaluates any policy which selects an action in $A_{s_t, t}^*$ for each $s_t \in S$ for all $t = 1, 2, \ldots, N - 1$.

**c.** It computes the expected total reward for the entire decision-making horizon, and from each period to the end of the horizon for any optimal policy.

Let $D_t^* \equiv \times_{s \in S} A_{s, t}^*$. Then any $\pi^* \in \Pi^* \equiv D_1^* \times \ldots \times D_{N-1}^*$ is an optimal policy. If more than one such $\pi^*$ exists, each yields the same expected total reward. This occurs if for some $s_t$, $A_{s_t, t}^*$ contains more than one action. To obtain a particular optimal policy, it is only necessary to retain a single action from $A_{s_t, t}^*$ for each $t \leq N - 1$ and $s_t \in S$.

Although this chapter emphasizes models with finite $S$ and $A$, the algorithm and results of Theorem 4.5.1 are valid in greater generality. It applies to models with countable, compact, or Polish state and action spaces. In nondiscrete models, regularity conditions are required to ensure that $u_t$ is measurable, integrals exist, and maxima are attained. Often we discretize the state and action spaces prior to computation; however, backward induction may be used to find optimal policies when the maxima and maximizing actions can be determined analytically. More importantly, a considerable portion of stochastic optimization literature uses backward induction to characterize the form of optimal policies under structural assumptions on rewards and transition probabilities. When such results can be obtained, specialized algorithms may be developed to determine the best policy of that type. We expand on this point in Sec. 4.7.

When there are $K$ states with $L$ actions in each, the backward induction algorithm requires $(N - 1)LK^2$ multiplications to evaluate and determine an optimal policy. Since there are $(L^K)^{(N-1)}$ deterministic Markovian policies, and direct evaluation of each requires $(N - 1)K^2$ multiplications, this represents considerable reduction in computation.

A further advantage of using backward induction is that, at pass $t$ through the algorithm, only $r_t$, $p_t$ and $u_{t+1}^*$ need be in high-speed memory. The data from iterations $t + 1$, $t + 2, \ldots, N$ are not required since they are summarized through $u_{t+1}^*$, and the data from decision epochs $1, 2, \ldots, t - 1$ are not needed until subsequent iterations. Of course, $A_{s, t}^*$ must be stored for all $t$ to recover all optimal policies.