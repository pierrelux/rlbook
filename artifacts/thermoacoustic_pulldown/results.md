| Experiment | Method | Status | Accepted updates | Cost | Energy (J) | Cold at 300 s (°C) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| Baseline pull-down | iLQR | converged | 24 | 51.224 | 1016 | 5.20 |
| Baseline pull-down | DDP | converged | 12 | 51.224 | 1016 | 5.20 |
| No parasitic load | iLQR | converged | 28 | 20.652 | 411 | 5.09 |
| No parasitic load | DDP | converged | 8 | 20.652 | 411 | 5.09 |
| Sluggish hot side | iLQR | converged | 28 | 45.716 | 911 | 5.13 |
| Sluggish hot side | DDP | converged | 14 | 45.585 | 908 | 5.13 |
| Amplitude-independent COP | iLQR | iteration_limit | 200 | 33.353 | 666 | 5.08 |
| Amplitude-independent COP | DDP | converged | 27 | 33.353 | 666 | 5.08 |
| Target out of reach in 300 s | iLQR | converged | 2 | 119.582 | 1426 | 2.20 |
| Target out of reach in 300 s | DDP | converged | 10 | 119.582 | 1426 | 2.20 |

Costs are evaluated on the final accepted nonlinear trajectory. The iteration-limit row is a retained local plan, not a converged solve.

For comparison, constant full amplitude in the baseline case ends at 2.20 °C, uses 1426 J, and has cost 149.89.
