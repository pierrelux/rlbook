# Semi-truck alley dock with single shooting

These deterministic single-shooting experiments support the sequential-methods section of `numerical-trajectory-optimization.md`. The tractor-semitrailer is a synthetic kinematic teaching model, not a measured vehicle.

Regenerate from the repository root with:

```bash
uv run python scripts/build_truck_parking_artifacts.py
```

The builder solves two scenarios with SciPy's L-BFGS-B on the reduced objective `J(u)`, whose value and reverse-mode gradient come from one compiled JAX rollout. Backing into the bay starts in the lane and ends docked; pulling out of the bay starts docked and ends in the lane. Both start from a straight, constant-speed seed. It writes:

- `metrics.json`: parameters, source hashes, solver settings, per-scenario stopping status, iteration and evaluation counts, arrival checks from a finer replay, steering-amplification summaries, and the final plans;
- `results.md`: the chapter's numerical tables;
- `../../_static/truck_parking/`: matching SVG, PDF, and PNG teaching figures (`iterations`, `paths`, `plan`, `conditioning`);
- `../../interactive/truck-parking-data.json`: checkpointed plans, the full cost trace, and the amplification profiles for the chapter's browser replay.

The state columns are the tractor rear-axle position in metres, the tractor heading, and the trailer heading in radians. The fifth wheel sits over the tractor's rear axle, and the trailer axle trails it by 10 m along the trailer axis. Controls are the tractor speed in m/s, bounded by ±1.5, and the steering angle in radians, bounded by ±30°. One RK4 step lasts 0.25 s, and the 160 control intervals cover a fixed 40 s horizon, so each solve has 320 decision variables.

The objective charges speed, steering, distance between the trailer axle and its target, a smooth penalty for any body corner within 0.2 m of the dock face, a smooth penalty for hitch angles beyond 60°, changes between consecutive controls, and a moving final interval. The terminal cost penalizes trailer-axle position error, trailer heading error, and hitch angle. The solver stops on a projected-gradient tolerance of 1e-5 or a relative objective decrease below 1e-10, with at most 1500 iterations.

Checkpointed plans keep every iterate up to iteration 10 and geometrically spaced iterates afterwards, plus the final one. The cost trace keeps every iteration.

Each final plan is replayed with four RK4 substeps per control interval. It must end with the trailer axle within 0.25 m of its target, the trailer heading within 2°, the hitch angle within 3°, the final speed below 0.1 m/s, every body corner above the dock face, the largest hitch angle below 60°, and controls within bounds. A stopping status other than `converged` or `small_decrease`, or a failed check, makes the builder exit with an error.

The amplification profiles divide the reverse-mode derivative of the terminal hitch angle with respect to each interval's steering angle by the tractor heading change that steering causes within the interval. They are compared against the product of the per-interval linearized hitch Jacobians along the same plan, for the backing plan and for the same plan driven forward by reversing the control sequence and the sign of the speed. Intervals with speed below 0.1 m/s are left undefined.

Single-run solver times exclude JAX compilation and vary by machine. They are diagnostic only. Ordinary book builds read these committed artifacts and do not run the optimizer.
