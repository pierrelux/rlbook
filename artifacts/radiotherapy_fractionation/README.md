# Radiotherapy fractionation with iLQR and DDP

These deterministic experiments support the fractionation section of `iterative-trajectory-optimization.md`. The phantom and the dose-response coefficients are teaching values, not clinical data. The problem is adapted from https://github.com/pierrelux/ddp-fractionation.

Regenerate from the repository root with:

```bash
uv run python scripts/build_radiotherapy_fractionation_artifacts.py
```

The builder solves the 20-session course with iLQR and DDP from the same uniform 2.1 Gy plan and evaluates the uniform equal-beam plan that just controls the tumor. It writes `metrics.json`, `results.md`, the figures in `../../_static/radiotherapy_fractionation/`, and the browser replay data `../../interactive/radiotherapy-fractionation-data.json`.

The state is the health of the tumor, spinal cord, heart, and lung. Controls are the three beam intensities, in gray at the tumor per session. The builder exits with an error if a solve does not converge, if accepted costs fail to decrease, or if the optimized plan leaves the tumor uncontrolled or an organ beyond tolerance.
