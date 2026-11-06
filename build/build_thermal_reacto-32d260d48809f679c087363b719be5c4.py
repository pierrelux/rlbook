#!/usr/bin/env python3
"""Explicit offline stages for the thermal-reactor prototype; never run by MyST."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import hashlib
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from thermal_reactor import (ReactorConfig, jnp, physics_step, physics_rollout,
                             ou_forecasts, field_metrics, valid_fields, warm_state)

DEFAULT_DATA = ROOT / "outputs" / "thermal_reactor" / "training"
DEFAULT_OUT = ROOT / "artifacts" / "thermal_reactor"


def generate(data_dir, episodes=256, steps=60, seed=20260911):
    data_dir = Path(data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    cfg = ReactorConfig()
    rng = np.random.default_rng(seed)
    fields = np.lib.format.open_memmap(data_dir / "fields.npy", mode="w+", dtype="float32",
                                      shape=(episodes, steps + 1, 2, cfg.ny, cfg.nx))
    controls = np.empty((episodes, steps, 4), np.float32)
    gusts = np.empty((episodes, steps), np.float32)
    yy, xx = np.meshgrid(np.linspace(0, 1, cfg.ny), np.linspace(0, 1, cfg.nx), indexing="ij")
    start = time.perf_counter()
    for offset in range(0, episodes, 16):
        n = min(16, episodes - offset)
        init = np.empty((n, 2, cfg.ny, cfg.nx), np.float32)
        init[:, 0] = 850.; init[:, 1] = 1.
        base = rng.uniform(.40, .53, (n, 4)).astype(np.float32)
        inlet_initial = rng.uniform(-25, 25, n).astype(np.float32)
        # Real PDE warm states, plus cold starts and smooth physical transients.
        initial = np.asarray(physics_step(init, base, inlet_initial, 200., cfg)).copy()
        for j in range(n):
            episode = offset + j
            if episode % 4 == 0:
                initial[j, 0] = rng.uniform(800, 900) + 15 * np.sin(np.pi * xx) * np.cos(np.pi * yy)
                initial[j, 1] = rng.uniform(.7, 1.)
            elif episode % 4 == 1:
                initial[j, 0] += 15 * np.exp(-((xx - .5)**2 + (yy - .3)**2) / .04)
            knots = base[j] + rng.normal(0, .065, (9, 4))
            knots = np.clip(knots, .20, .65)
            controls[episode] = np.stack([np.interp(np.arange(steps), np.linspace(0, steps - 1, 9), knots[:, a]) for a in range(4)], axis=-1)
            gusts[episode] = ou_forecasts(inlet_initial[j], steps, 1, rng, cfg)[:, 0]
        us = np.moveaxis(controls[offset:offset + n], 0, 1)
        gs = np.moveaxis(gusts[offset:offset + n], 0, 1)
        future = np.asarray(physics_rollout(jnp.asarray(initial), us, gs, cfg))
        if not valid_fields(future):
            raise RuntimeError("invalid reference trajectory during data generation")
        fields[offset:offset + n, 0] = initial
        fields[offset:offset + n, 1:] = np.moveaxis(future, 0, 1)
        fields.flush()
        print(f"generated {offset+n}/{episodes} episodes in {time.perf_counter()-start:.1f}s", flush=True)
    np.save(data_dir / "controls.npy", controls); np.save(data_dir / "disturbances.npy", gusts)
    permutation = np.random.default_rng(seed + 1).permutation(episodes)
    a, b = int(.7 * episodes), int(.85 * episodes)
    meta = {"config": cfg.to_dict(), "seed": seed, "steps": steps, "episodes": episodes,
            "train_episodes": permutation[:a].tolist(), "validation_episodes": permutation[a:b].tolist(),
            "test_episodes": permutation[b:].tolist(), "elapsed_s": time.perf_counter() - start}
    (data_dir / "dataset.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["generate", "train", "validate", "evaluate", "audit", "export", "plume", "recovery", "stirring", "all"])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--backend", choices=["jax", "metal"], default="jax")
    args = parser.parse_args()
    if args.stage == "stirring":
        from thermal_stirring_experiment import generate_stirring
        generate_stirring(ROOT,backend=args.backend)
        return
    if args.stage == "recovery":
        from thermal_recovery import generate_recovery
        generate_recovery(ROOT)
        return
    if args.stage == "plume":
        from thermal_plume import generate_plume
        generate_plume(ROOT)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("generate", "all"):
        generate(args.data_dir)
    if args.stage in ("train", "all"):
        from thermal_surrogate import train
        train(args.data_dir, args.output_dir, max_epochs=args.epochs)
    if args.stage in ("validate", "all"):
        from thermal_surrogate import Surrogate, validate_surrogate
        model, _ = Surrogate.load(args.output_dir / "surrogate.npz")
        from thermal_audit import audit_dataset
        audit_dataset(args.data_dir, args.output_dir)
        results = validate_surrogate(model, args.data_dir)
        results["checkpoint_sha256"] = hashlib.sha256((args.output_dir / "surrogate.npz").read_bytes()).hexdigest()
        results["inference_version"] = "residual-cnn-fraction-projection-v1"
        (args.output_dir / "surrogate_validation.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results["summary"], indent=2), flush=True)
    if args.stage in ("evaluate", "all"):
        from thermal_control import evaluate
        evaluate(args.output_dir, trials=args.trials, backend=args.backend)
    if args.stage in ("audit", "all"):
        from thermal_audit import audit
        audit(args.output_dir)
    if args.stage in ("export", "all"):
        from thermal_replay import export
        export(args.output_dir, ROOT)


if __name__ == "__main__":
    main()
