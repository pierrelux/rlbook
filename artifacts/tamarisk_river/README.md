# Managing tamarisk in a river network with value and policy iteration

These experiments support the tamarisk example in `infinite-horizon-mdps.md`. The MDP follows the invasive-species benchmark of Hall et al. (2012, Third International Conference on Computational Sustainability) and Taleghan et al. (2015, JMLR 16:3877–3903). Mortality (0.2), seeds per plant (100), the upstream and downstream dispersal parameters (0.1, 0.5), eradication success (0.85), planting success (0.65), action costs (0, 0.5, 0.9), damage costs (1.0 per invaded reach, 0.1 per tamarisk slot), mean exogenous seed arrivals (1 tamarisk, 4 native), the discount factor 0.9 and the budget of one treated reach per year are taken from Taleghan et al. The competitive advantage of tamarisk seeds (1.5) is one of the example values in Hall et al. The binary river tree, one slot per reach, the geometric dispersal kernel, Poisson seed rain and the order of events within a year are teaching choices: the papers delegate these details to a simulator. No field data are used.

Regenerate from the repository root with:

```bash
uv run python scripts/build_tamarisk_river_artifacts.py
```

The builder solves the budgeted MDP by value iteration for rivers of 1 to 8 reaches, by policy iteration up to 7 reaches, and the unbudgeted MDP (3^n joint actions) by value iteration up to 7 reaches, recording iterations and wall-clock seconds. For the 7-reach river it evaluates the optimal policy and four heuristics exactly, records 12 episodes of 40 years per starting state with common random numbers for every policy, and writes `metrics.json`, `results.md`, the figures in `../../_static/tamarisk_river/`, and the browser replay data `../../interactive/tamarisk-river-data.json`. It exits with an error if value and policy iteration disagree, if a budgeted heuristic beats the optimal policy in any state, or if the sampled episodes disagree grossly with the exact expected costs. Timings depend on the machine; the run takes about two minutes on four cores.
