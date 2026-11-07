| Fit | Targets per iteration | Fit error $\varepsilon$ | Bound $\varepsilon/(1-\gamma)$ | Value error | Bound $2\gamma\varepsilon/(1-\gamma)^2$ | Policy loss | Same package |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bilinear grid, step 2 | 45 | 0.79 | 40 | 2.24 | 3,884 | 0.098 | 96.5% |
| Bilinear grid, step 1 | 144 | 0.33 | 16 | 0.72 | 1,610 | 0.015 | 99.1% |
| Bilinear grid, step 0.5 | 527 | 0.09 | 4 | 0.18 | 420 | 0.005 | 99.5% |
| Extra trees | 2,000 | 0.76 | 38 | 1.77 | 3,744 | 0.258 | 93.6% |
| Neural network | 2,000 | 1.47 | 74 | 0.63 | 7,227 | 0.028 | 98.2% |

Fitted value iteration for 400 iterations with γ = 0.98. The fit error is the largest $|v_{k+1} - \mathrm{L} v_k|$ over the 12,231 states of the reference grid in the last 20 iterations. Value error and policy loss are measured against the reference solution, in millions of birds of discounted harvest; the optimal value ranges from about 30 to 51. *Same package* is the share of reference states where the greedy policy picks the reference package.
