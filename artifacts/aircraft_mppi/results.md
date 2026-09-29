| Controller | Fuel (kg) | Arrival error (m) | Feasible arrivals | Largest constraint violation |
|---|---:|---:|---:|---:|
| Mean-wind plan | 2577.8 | 66.2 | 1/1 | 0 |
| Frozen plan + gusts | 2577.8 | 2491.3 | 0/3 | 0 |
| Mean-wind replanning | 2587.0 | 102.9 | 3/3 | 1.15e-13 |
| Stochastic replanning | 2560.1 | 110.0 | 3/3 | 1.5e-13 |

The repeated runs use 3 independent gust seeds. The figures show mean and sample standard deviation across those seeds, with individual runs also plotted. The mean-wind plan is one deterministic reference run. Prediction uses 32 candidates, 6 shared wind scenarios, and 2 updates per decision. Arrival tolerances are 1000 m horizontally and 30 m vertically. Constraint residuals and replay differences are available in the downloadable metrics.

| Controller | Mean ESS | Total planning time per flight (s) | Rejected updates |
|---|---:|---:|---:|
| Frozen plan + gusts | 11.7 | 1.3 | 0 |
| Mean-wind replanning | 22.2 | 56.1 | 110 |
| Stochastic replanning | 8.8 | 52.2 | 92 |

Planning times are measured wall-clock times on the build machine; they include validation and vary with competing workloads. Rejected-update counts are totals across the recorded flights.
The largest 5 s versus 10 s replay difference is 0.261 m in position and 0.0020 kg in fuel. All vertical arrival errors are below 8.3e-13 m.
