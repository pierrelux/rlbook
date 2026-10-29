# Sailing to a windward mark in a shifting wind

These experiments support the sailing example in `infinite-horizon-mdps.md`. The model is Vanderbei's sailing problem (R. J. Vanderbei, *Sailing Strategies*, https://vanderbei.princeton.edu/sail/sail.html), used as the "sailing" benchmark by Péret and Garcia (ECAI 2004), Kocsis and Szepesvári (ECML 2006) and Bonet and Geffner (AAAI 2012).

Published values: leg times of 4, 3 and 2 minutes for close-hauled, beam-reach and broad-reach legs and a 3-minute tack delay (Vanderbei's page); 1 minute for a dead-downwind leg and the "shifty" wind transition matrix (the defaults in `sailing/sailing.h` of Bonet and Geffner's mdp-engine, https://github.com/bonetblai/mdp-engine, commit 0e614fa). The √2 factor on diagonal legs is our choice; it matches the cost range 1 to 8.6 reported by Kocsis and Szepesvári. Teaching values: the 21 × 21 lake, the start and mark positions, the "oscillating" wind chain, and the layline-sailor baseline. No measured sailing data are used.

Regenerate from the repository root with:

```bash
uv run python scripts/build_sailing_wind_shifts_artifacts.py
```

For three wind chains (steady, oscillating, shifty) the builder solves the undiscounted stochastic shortest-path problem by value iteration from v = 0, runs policy iteration from the layline sailor, evaluates the layline sailor and a certainty-equivalent planner exactly, and simulates 4000 paired races per wind chain. It writes `metrics.json`, `results.md`, the figure in `../../_static/sailing_wind_shifts/`, and the browser replay data `../../interactive/sailing-wind-shifts-data.json`. It exits with an error if the steady-wind optimum differs from the one-tack beat, if value and policy iteration disagree, if a baseline beats the optimal value anywhere, or if a Monte Carlo mean is more than four standard errors from the exact expected time.
