# Closing a storm-surge barrier by optimal stopping

These experiments support the storm-surge-barrier example in `stochastic-dp.md`. The setting is the Maeslant barrier (Maeslantkering) on the Nieuwe Waterweg near Rotterdam. Two facts are taken from published sources: the closing criterion, a forecast water level of 300 cm above NAP at Rotterdam (Rijkswaterstaat; Trace-Kleeberg et al. 2025, Coastal Engineering 200:104766), and the Port of Rotterdam's figure of a 1 in 100 failure probability per closure, which the closing-delay parameters are chosen to roughly match. Every other parameter is a teaching value: the forecast error (a Gaussian martingale with standard deviation 4 cm per square-root hour), the 4-hour warning period and 2-hour closing operation, the fault probability (0.1) and hourly repair probability (0.7), the 6 hours of halted shipping after the peak, the storm distribution (peak forecast 24 h ahead normal with mean 280 cm and standard deviation 20 cm), and the flood damage measured in hours of halted shipping. No measured forecast or water-level data are used.

Regenerate from the repository root with:

```bash
uv run python scripts/build_storm_surge_barrier_artifacts.py
```

For five values of the flood damage, the builder solves the finite-horizon MDP by backward induction, evaluates the optimal policy, the best rule of the form "order the closure once the forecast reaches a fixed level, between 6 h and a latest lead before the peak" (found by exhaustive search over levels and leads), and the closing criterion applied as soon as it is reached, all exactly by backward policy evaluation. It writes `metrics.json`, `results.md`, the figure in `../../_static/storm_surge_barrier/`, and the browser replay data `../../interactive/storm-surge-barrier-data.json` (eight storms drawn from the model with a fixed seed). It exits with an error if the optimal policy is worse than a threshold rule, if its closing region is not an upper set of forecasts at every lead, if it orders a closure that can no longer finish, or if a Monte Carlo average over 4000 sampled storms disagrees with the exact expected costs.
