| $n$ | Chebyshev: value error | fishes out below | policy loss | Piecewise linear: value error | fishes out below | policy loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 3.47 | 0.29 | 0.647 | 0.96 | 0.00 | 0.197 |
| 10 | diverged |  |  | 0.64 | 0.19 | 0.049 |
| 12 | 1.66 | 0.31 | 1.180 | 0.52 | 0.17 | 0.064 |
| 20 | 1.04 | 0.27 | 0.357 | 0.22 | 0.20 | 0.041 |
| 40 | 0.03 | 0.00 | 0.006 | 0.09 | 0.22 | 0.014 |

Collocation by function iteration with $n$ coefficients. *Value error* is $\max_x |\hat v(x) - v^\star(x)|$; the optimal value ranges from 0 to 4.5. *Fishes out below* is the largest stock that the greedy policy of $\hat v$ harvests to zero; the optimal policy fishes out stocks below 0.23, and the tipping point is at 0.3. *Policy loss* is $\max_x (v^\star - v^{\hat\pi})(x)$, with the greedy policy $\hat\pi$ evaluated exactly. *Diverged*: the coefficients grew without bound.
