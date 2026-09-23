# Thermoacoustic pull-down with iLQR and DDP

These deterministic local trajectory-optimization experiments support `iterative-trajectory-optimization.md`. The refrigerator is a synthetic teaching model, not a measured device.

Regenerate from the repository root with:

```bash
uv run python scripts/build_thermoacoustic_pulldown_artifacts.py
```

The builder solves five variants with both iLQR and DDP from the same constant-amplitude seed. It also evaluates a constant full-amplitude plan and two other baseline seeds. It writes:

- `metrics.json`: parameters, source hashes, solver settings, final plans, accepted costs, energy, status, and replay checks;
- `results.md`: the chapter's numerical comparison;
- `../../_static/thermoacoustic_pulldown/`: matching SVG, PDF, and PNG teaching figures.

The state columns are cold and hot temperatures in degrees Celsius. Controls are driver-amplitude fractions in [0, 1]. One RK4 step lasts 1 s, and the 300 controls cover a fixed 300 s horizon. Energy is the sum of acoustic work over those intervals. The running cost weights that energy, and the terminal cost penalizes the final cold-temperature error.

Each final plan is checked for finite states, control bounds, and agreement with a four-times-finer RK4 replay. Reachable-target runs must end within 0.6 K of their targets. The 300 s out-of-reach variant still undergoes bounds and replay checks. All accepted costs must decrease. The no-minor-loss iLQR run intentionally retains its last accepted plan at the iteration limit; its DDP counterpart converges. An unexpected status or failed check causes the builder to exit with an error after writing its diagnostic reports.

Single-run solver times exclude JAX compilation and vary by machine. They are diagnostic only. Ordinary book builds read these committed artifacts and do not run the optimizer.
