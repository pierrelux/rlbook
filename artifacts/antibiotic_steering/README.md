# Steering antibiotic resistance with backward induction

These experiments support the antibiotic-steering example in `stochastic-dp.md`. The fitness landscapes are the growth rates measured by Mira et al. (2015, PLoS ONE 10:e0122283) for the 16 TEM β-lactamase genotypes of *E. coli* under 15 β-lactams. The evolutionary model is the SSWM Markov chain of Nichol et al. (2015, PLoS Comput. Biol. 11:e1004493) with r = 0, adapted from https://github.com/Daniel-Nichol/EvolutionarySteering (MIT licence).

Regenerate from the repository root with:

```bash
uv run python scripts/build_antibiotic_steering_artifacts.py
```

For every final drug, with zero to three steering drugs and two starting populations (uniform over genotypes, and the wild type), the builder finds the best fixed drug sequence by exhaustive search and the Markov policy by backward induction. It writes `metrics.json`, `results.md`, the figure in `../../_static/antibiotic_steering/`, and the browser replay data `../../interactive/antibiotic-steering-data.json`. It exits with an error if the feedback policy is ever worse than the best fixed sequence, and checks that each replayed course ends with the probability the solver reported.
