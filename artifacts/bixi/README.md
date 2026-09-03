# BIXI textbook artifacts

`textbook_results.json` contains the recorded showcase trajectories, frozen
open-loop plan, deterministic mean-flow calculation, 512-seed aggregate
summaries, and censoring counterexample used in the chapter. The corresponding
per-seed metrics are in `controller_metrics.csv`. `results.md` is a generated
MyST-ready fragment, so prose that quotes the main means cannot drift from the
JSON.

The JSON is the numerical source for both the inline replay and the committed
SVG/PDF fallbacks. A normal book build does not regenerate or resample it.
`manifest.json` records input and output checksums.

Rebuild after preparing `data/bixi`:

```bash
uv run python scripts/build_bixi_artifacts.py --seeds 512
```

These are results for a transparent teaching model calibrated from completed
trip counts. They are not estimates of failures or relocation performance in
BIXI's operated network.
