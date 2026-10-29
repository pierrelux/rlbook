# SwingRL PPO experiment

These files record the PPO experiment discussed in `pg.md`. The experiment
uses the articulated standing model from SwingRL commit
`d579663fc81c044729f4d3ab60bf63bcdbd27b9a` and defines success as one full
unwrapped rotation.

The prespecified protocol trained seeds 0 through 4 for 1,000,000 requested
environment interactions per seed. A rollout contains 256 transitions from
each of eight batched environments, so the last complete update occurs at
1,001,472 interactions. Checkpoints are the first complete update at or beyond
each 50,000-interaction target. The checkpoint CSV files record both counts.

None of the five PPO policies completed a rotation on the 100 fixed held-out
initial states. Their mean final return was -5.11. The structured phase-locked
controller succeeded from all 100 states with mean return 191.79, but its
nominal rigid-link trajectories required negative suspension tension during
7.37 percent of active steps. A chain cannot supply that force.

Each `seed_*` directory contains:

- `manifest.json`, with the complete configuration, software revisions,
  hardware, evaluation distribution, and elapsed time;
- `updates.csv`, with every PPO update and optimization diagnostic;
- `checkpoints.csv`, with fixed held-out evaluations;
- `checkpoints/*.npz`, with policy and value-network parameters;
- `showcase/*.npz`, with the fixed-start deterministic rollout trace saved at
  each checkpoint.

The complete experiment and the Python-rendered replay can be regenerated from
the repository root with:

```bash
uv run python code/swing_ppo.py all \
  --artifacts artifacts/swing_ppo \
  --output _static/swing_ppo \
  --seeds 0 1 2 3 4 \
  --steps 1000000
```

The MyST build reads the committed outputs in `_static/swing_ppo`; it does not
train a policy.

When only the shared analytical baseline changes, refresh its metrics and the
derived comparison figure without loading or modifying any PPO checkpoint:

```bash
uv run python code/swing_ppo.py refresh-baseline \
  --artifacts artifacts/swing_ppo \
  --output _static/swing_ppo \
  --seeds 0 1 2 3 4
```
