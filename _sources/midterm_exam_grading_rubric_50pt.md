# IFT6162 Midterm Exam - Grading Rubric

**Total: 60 points | Duration: 2 hours**

---

## Question 1: Sequential vs. Simultaneous — **6 points**

### Part (a): Identification + Structural Difference [4 pts]
- Identifies both snippets correctly: **2 pts**
- Explains structural difference: **2 pts**

### Part (b): State Bounds [2 pts]
- Correct answer with brief explanation: **2 pts**

---

## Question 2: COCP to DOCP — **15 points**

### Part (a): Explicit Euler ★ FUNDAMENTAL [6 pts]
- Correct discrete dynamics $x_{k+1} = x_k + \Delta t \cdot f(x_k, u_k, t_k)$: **6 pts**
  - General form: 3 pts
  - Substituted correctly: 2 pts
  - Initial condition: 1 pt

### Part (b): Integral Cost [3 pts]
- Riemann sum with $\Delta t$: **2 pts**
- Correct stage cost: **1 pt**

### Part (c): Path Constraints [2 pt]
- Correct indices for bounds: **1 pt** 1pt why

### Part (d): Control Synthesis [2 pt]
- Valid interpolation method: **1 pt**1pt why

### Part (e): Problem Structure [2 pt]
- Sparse structure identified: **1 pt**1pt why (linear, quadratic)

---

## Question 3: Short-Horizon (MPC) — **8 points**

### Part (a): Receding Horizon Strategy [3 pts]
- Solve-apply-repeat cycle: **3 pts**

### Part (b): Robustness [3 pts]
- Feedback property: **2 pts**
- Handles uncertainty: **1 pt**

### Part (c): Terminal Cost [2 pts]
- Mentions terminal cost/penalty: **2 pts**

---

## Question 4: Bellman Operator — **17 points**

### Part (a): Operator Signature [2 pts]
- Correct signature $\mathrm{L}: (\mathcal{S} \to \mathbb{R}) \to (\mathcal{S} \to \mathbb{R})$: **2 pts**

### Part (b): Composition & L_π Matrix ★ FUNDAMENTAL [4 pts]
- (i) $\mathrm{L}(\mathrm{L}v)$ explained: **1 pt**
- (ii) $\mathrm{L}_\pi$ as matrix $\mathbf{r}_\pi + \gamma \mathbf{P}_\pi v$: **3 pts**
  - This is fundamental - everyone should know policy evaluation in vector form

### Part (c): Implementation Types [3 pt]
- Correct categorization: **1 pt** each 

### Part (d): Exponential Complexity [2 pt]
- Identifies I with explanation: **2 pts**

### Part (e): Lazy vs Eager [2 pt]
- Correct: **1 pt**, why 1 pt

### Part (f): Greedy Selector Equivalence [2 pt]
- Key insight mentioned: **1 pt**, why 1 pt

### Part (g): Faithful vs Practical [2 pts]
why 1pt

---

## Question 5: Projection Methods — **14 points**

### Part (a): Collocation System [6 pts]
- Correct system: $\hat{v}(s_k) = (\text{Bellman RHS at } s_k)$ for $k=1,\ldots,n$: **6 pts**
  - Structure correct: 3 pts
  - Equation form correct: 3 pts

### Part (b): Galerkin & Quadrature [8 pts]
- (i) Identifies numerical technique: **6 pts**
  - **Quadrature**: 6 pts
  - **Monte Carlo**: 6 pts ★ EQUALLY VALID
  - Vague "discretization": 3 pts
- (ii) Why collocation avoids this: **2 pts**

---

## Grading Summary

**Total: 60 points**

| Question | Points | % of Exam | Difficulty |
|----------|--------|-----------|------------|
| Q1 | 6 | 10% | Easy |
| Q2 | 15 | 25% | Moderate |
| Q3 | 8 | 13% | Easy |
| Q4 | 17 | 28% | Moderate |
| Q5 | 14 | 23% | Hard |

**Fundamental concepts (weighted heavily):**
- Q2 Part (a): Explicit Euler (6 pts) - Core discretization skill
- Q4 Part (b): $\mathrm{L}_\pi$ matrix form (4 pts) - Policy evaluation is foundational
- Q5 Part (b): Quadrature/MC (6 pts) - Both methods equally valid

**Grade Conversion:**
- **A (≥51):** Excellent (≥85%)
- **B (42-50):** Good (70-84%)
- **C (30-41):** Adequate (50-69%)
- **D (21-29):** Minimal (35-49%)
- **F (<21):** Insufficient (<35%)

**Grading Efficiency:**
- Whole numbers only
- Award 0, 1, or full points for most parts
- Q2(a) and Q4(b): binary (right or wrong, heavy weight)
- Fast grading: ~15-20 min per exam

**Common Deductions:**
- Q2(a): Missing $\Delta t$ (-2 pts)
- Q4(b): Can't write matrix form (-3 pts)
- Q5(b): Doesn't recognize quadrature/MC (-6 pts major)


