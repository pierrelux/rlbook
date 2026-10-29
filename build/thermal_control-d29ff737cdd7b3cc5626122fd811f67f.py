"""Surrogate MPPI, PDE acceptance checks, and paired thermal-reactor studies."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import partial
import json
import hashlib
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import expit, logit
from scipy.stats import t as student_t

from mppi_control import gaussian_mppi_update
from thermal_reactor import (ReactorConfig, physics_step, physics_rollout, warm_state,
                             field_metrics, ou_forecasts, random_streams, valid_fields)
from thermal_surrogate import Surrogate, predict


@dataclass(frozen=True)
class PlannerConfig:
    horizon: int = 20
    knots: int = 6
    candidates: int = 128
    ensemble: int = 4
    iterations: int = 2
    variance: float = .7**2
    temperature: float = .025
    baseline: float = .48
    latent_scale: float = .35
    quality_weight: float = 5.
    heat_weight: float = 10.
    slew_weight: float = .05


def controls_from_latent(z, planner=PlannerConfig()):
    """Bounded physical controls; Gaussian densities live on z, not controls."""
    z = np.asarray(z)
    knots = expit(logit(planner.baseline) + planner.latent_scale * z)
    at = np.linspace(0, planner.horizon - 1, planner.knots)
    # Output (..., horizon, four controls).
    flat = knots.reshape(-1, planner.knots, 4)
    result = np.stack([np.stack([np.interp(np.arange(planner.horizon), at, row[:, a])
                                for a in range(4)], axis=-1) for row in flat])
    return result.reshape(z.shape[:-2] + (planner.horizon, 4)).astype(np.float32)


def shifted_latent(tape, planner=PlannerConfig()):
    shifted = np.concatenate([tape[1:], tape[-1:]], axis=0)
    at = np.linspace(0, planner.horizon - 1, planner.knots)
    knots = np.stack([np.interp(at, np.arange(planner.horizon), shifted[:, a]) for a in range(4)], axis=-1)
    return (logit(np.clip(knots, 1e-6, 1 - 1e-6)) - logit(planner.baseline)) / planner.latent_scale


@partial(jax.jit, static_argnames=("config", "planner", "physics"))
def rollout_scores(params, norm, initial, tapes, forecasts, previous,
                   config=ReactorConfig(), planner=PlannerConfig(), physics=False):
    """Candidate x scenario batches; reduce costs before applying any weights."""
    K, H, _ = tapes.shape
    L = forecasts.shape[1]
    x = jnp.broadcast_to(initial, (K, L) + initial.shape).reshape((-1,) + initial.shape)
    u = jnp.repeat(jnp.swapaxes(tapes, 0, 1), L, axis=1)
    g = jnp.broadcast_to(forecasts[:, None, :], (H, K, L)).reshape((H, K * L))
    last = jnp.broadcast_to(previous, (K * L, 4))

    def advance(carry, ug):
        state, old_u, accumulated = carry
        ctrl, gust = ug
        if physics:
            following = physics_step(state, ctrl, gust, config.control_dt, config)
        else:
            following = predict(params, norm, state, ctrl, gust, config)
        peak = jnp.max(following[:, 0], axis=(-2, -1))
        conversion = 1 - jnp.mean(following[:, 1, :, -1], axis=-1)
        cost = (jnp.mean(ctrl, axis=-1)
                + planner.quality_weight * (jnp.maximum(config.conversion_target - conversion, 0) / .05)**2
                + planner.heat_weight * (jnp.maximum(peak - config.temperature_limit, 0) / 20)**2
                + planner.slew_weight * jnp.mean(((ctrl - old_u) / .1)**2, axis=-1))
        bad = (~jnp.all(jnp.isfinite(following), axis=(1, 2, 3))
               | (jnp.min(following[:, 0], axis=(-2, -1)) < 300)
               | (jnp.max(following[:, 0], axis=(-2, -1)) > 1600)
               | (jnp.min(following[:, 1], axis=(-2, -1)) < -.05)
               | (jnp.max(following[:, 1], axis=(-2, -1)) > 1.05))
        cost = jnp.where(bad, jnp.inf, cost)
        return (following, ctrl, accumulated + cost / H), None

    total = jax.lax.scan(advance, (x, last, jnp.zeros(K * L)), (u, g))[0][2]
    return jnp.mean(total.reshape(K, L), axis=-1)


def check_plan(initial, tape, current_gust, config=ReactorConfig()):
    # Conditional mean includes the observed inlet in the first interval.
    forecast = ou_forecasts(current_gust, len(tape), 1, np.random.default_rng(0), config, mean_only=True)[:, 0]
    future = np.asarray(physics_rollout(initial, tape, forecast, config))
    metrics = field_metrics(future)
    feasible = (valid_fields(future)
                and metrics["peak_temperature_K"].max() <= config.temperature_limit + 1e-4
                and metrics["outlet_conversion"].min() >= config.conversion_target - 1e-6)
    return bool(feasible), future


class MPPIController:
    def __init__(self, model, planner=PlannerConfig(), seed=0, stochastic=True, physics=False):
        self.model, self.config, self.planner = model, model.config, planner
        _, self.proposal_rng, self.forecast_rng = random_streams(seed)
        self.stochastic, self.physics = stochastic, physics
        self.mean = np.zeros((planner.knots, 4))
        self.previous = np.full(4, planner.baseline)
        self.last_tape = None
        self.accelerator = None
        if getattr(model, "backend", "jax") == "metal" and not physics:
            from thermal_metal import MetalScorer
            self.accelerator = MetalScorer(model, planner)

    def score(self, x, tapes, forecasts):
        if self.accelerator is not None:
            return self.accelerator.score(x, tapes, forecasts, self.previous)
        # Fixed-size chunks keep the candidate/scenario batch within CPU memory.
        values = []
        for first in range(0, len(tapes), 32):
            block = tapes[first:first + 32]
            count = len(block)
            if count < 32:
                block = np.concatenate([block, np.repeat(block[-1:], 32 - count, axis=0)])
            values.extend(np.asarray(rollout_scores(self.model.params, self.model.norm, jnp.asarray(x),
                                                    jnp.asarray(block), jnp.asarray(forecasts), jnp.asarray(self.previous),
                                                    self.config, self.planner, self.physics))[:count])
        return np.asarray(values)

    def plan(self, x, current_gust, capture=False):
        p = self.planner; start = time.perf_counter()
        L = p.ensemble if self.stochastic else 1
        forecast = ou_forecasts(current_gust, p.horizon, L, self.forecast_rng, self.config,
                                mean_only=not self.stochastic)
        tape = controls_from_latent(self.mean, p)
        feasible, _ = check_plan(x, tape, current_gust, self.config)
        if not feasible:
            # A fixed reference is checked from the CURRENT measured state.
            candidate = np.full_like(tape, p.baseline)
            feasible, _ = check_plan(x, candidate, current_gust, self.config)
            if feasible:
                self.mean = np.zeros_like(self.mean); tape = candidate
        record = {"accepted": 0, "rejected": 0, "planning_failure": not feasible,
                  "ess": [], "current_gust_K": float(current_gust), "forecast_ensemble": L,
                  "invalid_candidates": 0}
        if not feasible:
            record["runtime_s"] = time.perf_counter() - start
            # Mark failure; the episode ends before an unvalidated action is used.
            return None, record, None
        incumbent_cost = self.score(x, tape[None], forecast)[0]
        snapshot = None
        for iteration in range(p.iterations):
            samples = self.proposal_rng.normal(size=(p.candidates, p.knots, 4)) * np.sqrt(p.variance) + self.mean
            tapes = controls_from_latent(samples, p)
            costs = self.score(x, tapes, forecast)
            record["invalid_candidates"] += int(np.sum(~np.isfinite(costs)))
            try:
                proposed, weights, ess = gaussian_mppi_update(samples, costs, self.mean,
                                                              p.variance, p.temperature)
            except ValueError:
                record["rejected"] += 1
                continue
            proposed_tape = controls_from_latent(proposed, p)
            proposed_cost = self.score(x, proposed_tape[None], forecast)[0]
            valid, proposed_fields = check_plan(x, proposed_tape, current_gust, self.config)
            accepted = bool(valid and np.isfinite(proposed_cost) and proposed_cost < incumbent_cost)
            record["ess"].append(float(ess))
            if capture and iteration == p.iterations - 1:
                # Highest weights plus representative ranks, with actual weights
                # retained from the ENTIRE candidate batch (never renormalized).
                order = np.argsort(-weights)
                selected = order[[0, 1, 2, p.candidates // 4, p.candidates // 2, p.candidates - 1]]
                common_g = forecast[:, 0]
                init = jnp.broadcast_to(x, (6,) + x.shape)
                candidate_fields = np.asarray(self.model.rollout(init, np.swapaxes(tapes[selected], 0, 1),
                                                                 np.repeat(common_g[:, None], 6, axis=1)))
                predicted_update = np.asarray(self.model.rollout(x, proposed_tape, common_g))
                pde_update = np.asarray(physics_rollout(x, proposed_tape, common_g, self.config))
                snapshot = {"initial": np.asarray(x), "candidate_fields": candidate_fields,
                            "candidate_controls": tapes[selected], "candidate_costs": costs[selected],
                            "candidate_weights": weights[selected], "candidate_indices": selected,
                            "proposed_controls": proposed_tape, "proposed_surrogate": predicted_update,
                            "proposed_pde": pde_update, "forecast": common_g,
                            "full_forecasts": forecast, "proposed_accepted": accepted,
                            "proposed_mean_feasible": valid, "ess": float(ess),
                            "proposed_cost": float(proposed_cost), "incumbent_cost": float(incumbent_cost)}
            if accepted:
                self.mean, tape, incumbent_cost = proposed, proposed_tape, proposed_cost
                record["accepted"] += 1
            else:
                record["rejected"] += 1
        chosen = tape[0].copy()
        self.last_tape = tape.copy()
        self.mean = shifted_latent(tape, p)
        self.previous = chosen
        record.update({"runtime_s": time.perf_counter() - start, "forecast_cost": float(incumbent_cost) if np.isfinite(incumbent_cost) else None})
        return chosen, record, snapshot


def paired_interval(values):
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "mean": float(a.mean()),
            "ci95_halfwidth": float(student_t.ppf(.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None}


def evaluate(output_dir, trials=8, backend="jax"):
    output_dir = Path(output_dir)
    model, metadata = Surrogate.load(output_dir / "surrogate.npz")
    model.backend = backend
    cfg, planner = model.config, PlannerConfig()
    initial = warm_state(cfg, planner.baseline)
    if not check_plan(initial, np.full((planner.horizon, 4), planner.baseline), 0., cfg)[0]:
        raise RuntimeError("constant-heating reference is infeasible before evaluation")
    validation = json.loads((output_dir / "surrogate_validation.json").read_text())
    fingerprint = hashlib.sha256((output_dir / "surrogate.npz").read_bytes()).hexdigest()
    if validation.get("checkpoint_sha256") != fingerprint:
        raise RuntimeError("Held-out validation must match the current checkpoint; run validate first")
    provenance = {"checkpoint_sha256": fingerprint, "planner": asdict(planner),
                  "inference_version": "residual-cnn-fraction-projection-v1", "backend": backend}
    status = "passed held-out surrogate criteria" if validation["summary"]["passed"] else "experimental: surrogate failed held-out criteria"
    rows, decisions = [], {}
    for trial in range(trials):
        physical_rng, _, _ = random_streams(90000 + trial)
        gust = ou_forecasts(0., 80, 1, physical_rng, cfg)[:, 0]
        for name in ("constant", "mean", "stochastic"):
            path = output_dir / f"trial-{trial}-{name}.npz"
            # Resume expensive offline runs, keeping one checkpoint per episode.
            if path.exists():
                with np.load(path) as saved:
                    if json.loads(str(saved["provenance"])) != provenance:
                        raise RuntimeError(f"Stale experiment checkpoint: {path}; use a new output directory")
                    rows.append(json.loads(str(saved["metrics"]))); decisions[f"{trial}-{name}"] = json.loads(str(saved["decisions"]))
                print(f"reuse trial {trial} {name}", flush=True)
                continue
            x = initial.copy(); fields = [x]; controls = []; records = []
            controller = MPPIController(model, planner, seed=190000 + trial, stochastic=name == "stochastic")
            for k in range(80):
                if name == "constant":
                    u, rec, snap = np.full(4, planner.baseline), {"runtime_s": 0., "accepted": 0, "rejected": 0, "ess": [], "planning_failure": False}, None
                else:
                    u, rec, snap = controller.plan(x, float(gust[k]), capture=(trial == 0 and k in (0, 20, 40)))
                records.append(rec)
                if u is None:
                    break
                if snap is not None:
                    np.savez_compressed(output_dir / f"candidates-{name}-{k}.npz", **snap)
                controls.append(u)
                x = np.asarray(physics_step(x, u, gust[k], cfg.control_dt, cfg))
                if not valid_fields(x):
                    raise RuntimeError("invalid executed PDE state")
                fields.append(x)
                if k % 10 == 0:
                    print(f"trial {trial} {name} step {k}: {field_metrics(x)}", flush=True)
            fields, controls = np.asarray(fields), np.asarray(controls).reshape(-1, 4)
            fm = field_metrics(fields)
            all_ess = [e for r in records for e in r["ess"]]
            row = {"trial": trial, "controller": name, "steps_completed": len(controls),
                   "completed": len(controls) == 80,
                   "normalized_heating_energy_s": float(controls.mean(axis=1).sum() * cfg.control_dt),
                   "minimum_outlet_conversion": float(fm["outlet_conversion"].min()),
                   "maximum_temperature_K": float(fm["peak_temperature_K"].max()),
                   "quality_violation_duration_s": float(np.sum(fm["outlet_conversion"][1:] < cfg.conversion_target) * cfg.control_dt),
                   "temperature_violation_duration_s": float(np.sum(fm["peak_temperature_K"][1:] > cfg.temperature_limit) * cfg.control_dt),
                   "planning_runtime_s": float(sum(r["runtime_s"] for r in records)),
                   "accepted": sum(r["accepted"] for r in records), "rejected": sum(r["rejected"] for r in records),
                   "mean_ess": float(np.mean(all_ess)) if all_ess else None,
                   "planning_failures": sum(r["planning_failure"] for r in records)}
            np.savez_compressed(path, fields=fields, controls=controls, disturbances=gust,
                                metrics=np.array(json.dumps(row)), decisions=np.array(json.dumps(records)),
                                provenance=np.array(json.dumps(provenance)))
            rows.append(row); decisions[f"{trial}-{name}"] = records
            print(json.dumps(row), flush=True)
    pairs = []
    for trial in range(trials):
        by_method = {r["controller"]: r for r in rows if r["trial"] == trial}
        if all(by_method[m]["completed"] for m in ("mean", "stochastic")):
            pairs.append(by_method["stochastic"]["normalized_heating_energy_s"] - by_method["mean"]["normalized_heating_energy_s"])
    summary = {"config": cfg.to_dict(), "planner": asdict(planner), "surrogate_status": status,
               "surrogate_checkpoint": metadata, "provenance": provenance, "inference_backend": backend, "rows": rows,
               "paired_energy_stochastic_minus_mean": paired_interval(pairs) if pairs else None,
               "paired_energy_scope": "Only pairs completing all 160 s; incomplete trials are reported separately.",
               "seeds": {"physical": list(range(90000, 90000 + trials)), "planning": list(range(190000, 190000 + trials))}}
    (output_dir / "evaluation.json").write_text(json.dumps(summary, indent=2))
    sensitivity(model, initial, output_dir)


def sensitivity(model, initial, output_dir):
    rows = []
    for K, L in ((32, 4), (128, 4), (512, 4), (128, 1), (128, 16)):
        for seed in range(3):
            planner = PlannerConfig(candidates=K, ensemble=L)
            ctl = MPPIController(model, planner, seed=300000 + seed)
            # Initial decision, full same PDE-replayed control sequence.
            u, rec, _ = ctl.plan(initial, 0.)
            row = {"candidates": K, "ensemble": L, "seed": seed, **rec}
            if u is not None:
                row["first_control"] = u.tolist()
                held_out = ou_forecasts(0., planner.horizon, 32, np.random.default_rng(500000), model.config)
                row["held_out_pde_cost"] = float(np.asarray(rollout_scores(
                    model.params, model.norm, initial, ctl.last_tape[None], held_out,
                    np.full(4, planner.baseline), model.config, planner, True))[0])
            rows.append(row)
            print(f"sensitivity {K} x {L}, seed {seed}", flush=True)
    comparison = []
    for physics in (False, True):
        ctl = MPPIController(model, PlannerConfig(), seed=400000, physics=physics)
        u, rec, _ = ctl.plan(initial, 0., capture=True)
        result = {"rollout_model": "PDE" if physics else "surrogate", "first_control": None if u is None else u.tolist(), **rec}
        if u is not None:
            held_out = ou_forecasts(0., 20, 32, np.random.default_rng(500000), model.config)
            result["held_out_pde_cost"] = float(np.asarray(rollout_scores(
                model.params, model.norm, initial, ctl.last_tape[None], held_out,
                np.full(4, .48), model.config, PlannerConfig(), True))[0])
        comparison.append(result)
    (Path(output_dir) / "sensitivity.json").write_text(json.dumps({"sampling": rows, "model_comparison": comparison, "timing_note": "Local wall-clock planning times include PDE validation and first-use compilation where needed. The model-comparison calls also capture snapshots; sampling-count calls do not."}, indent=2))
