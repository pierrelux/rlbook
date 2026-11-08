"""JAX residual field surrogate and trajectory-separated training utilities."""
from __future__ import annotations

from functools import partial
import json
import hashlib
import platform
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

from thermal_reactor import ReactorConfig, heating_field, inlet_profile


DILATIONS = (1, 2, 4, 1)


def init_params(seed=1234):
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    sizes = (6, 16, 16, 16, 2)
    return tuple({"w": (jax.random.normal(keys[i], (3, 3, sizes[i], sizes[i + 1]))
                        * np.sqrt(2 / (9 * sizes[i])) if i < 3 else
                        jnp.zeros((3, 3, sizes[i], sizes[i + 1]))),
                  "b": jnp.zeros(sizes[i + 1])} for i in range(4))


@partial(jax.jit, static_argnames=("config", "bound_fraction"))
def predict(params, norm, fields, controls, disturbance, config=ReactorConfig(), bound_fraction=True):
    """A full 2-second transition, (...,2,ny,nx) -> same physical-unit shape.

    The residual network is trained without output constraints. At inference,
    reactant fraction is projected onto [0, 1]; temperature is unrestricted.
    Held-out validation also reports raw predictions and projection magnitude.
    """
    x = jnp.asarray(fields, dtype=jnp.float32)
    leading = x.shape[:-3]
    x = x.reshape((-1, 2, config.ny, config.nx))
    u = jnp.asarray(controls, dtype=jnp.float32).reshape((-1, 4))
    g = jnp.asarray(disturbance, dtype=jnp.float32).reshape((-1,))
    scaled = (x - norm["mean"][None, :, None, None]) / norm["std"][None, :, None, None]
    h = heating_field(u, config)[:, None]
    inlet = jnp.broadcast_to((inlet_profile(g, config) - norm["mean"][0])[:, None, :, None]
                             / norm["std"][0], (len(x), 1, config.ny, config.nx))
    yy, xx = jnp.meshgrid(jnp.linspace(-1, 1, config.ny), jnp.linspace(-1, 1, config.nx), indexing="ij")
    coordinates = jnp.broadcast_to(jnp.stack([xx, yy])[None], (len(x), 2, config.ny, config.nx))
    z = jnp.moveaxis(jnp.concatenate([scaled, h, inlet, coordinates], axis=1), 1, -1)
    for i, (layer, dilation) in enumerate(zip(params, DILATIONS)):
        # Edge padding avoids an artificial zero-temperature exterior.
        padded = jnp.pad(z, ((0, 0), (dilation, dilation), (dilation, dilation), (0, 0)), mode="edge")
        z = jax.lax.conv_general_dilated(padded, layer["w"], (1, 1), "VALID",
                                        rhs_dilation=(dilation, dilation),
                                        dimension_numbers=("NHWC", "HWIO", "NHWC")) + layer["b"]
        if i < 3:
            z = jax.nn.silu(z)
    out = x + jnp.moveaxis(z, -1, 1) * norm["std"][None, :, None, None]
    if bound_fraction:
        out = out.at[:, 1].set(jnp.clip(out[:, 1], 0., 1.))
    return out.reshape(leading + (2, config.ny, config.nx))


@partial(jax.jit, static_argnames=("config", "bound_fraction"))
def surrogate_rollout(params, norm, initial, controls, disturbances, config=ReactorConfig(), bound_fraction=True):
    def advance(x, ug):
        y = predict(params, norm, x, ug[0], ug[1], config, bound_fraction)
        return y, y
    return jax.lax.scan(advance, initial, (controls, disturbances))[1]


class Surrogate:
    def __init__(self, params, norm, config=ReactorConfig()):
        self.params, self.norm, self.config = params, norm, config

    def step(self, fields, controls, disturbance, elapsed=2.0):
        if abs(elapsed - self.config.control_dt) > 1e-8:
            raise ValueError("surrogate supports only its trained control interval")
        return predict(self.params, self.norm, fields, controls, disturbance, self.config)

    def rollout(self, initial, controls, disturbances):
        return surrogate_rollout(self.params, self.norm, initial, controls, disturbances, self.config)

    def save(self, path, metadata=None):
        path = Path(path)
        arrays = {f"layer_{i}_{key}": np.asarray(value) for i, layer in enumerate(self.params)
                  for key, value in layer.items()}
        arrays.update({"mean": np.asarray(self.norm["mean"]), "std": np.asarray(self.norm["std"])})
        arrays["metadata"] = np.array(json.dumps({"config": self.config.to_dict(), **(metadata or {})}))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as a:
            params = tuple({key: jnp.asarray(a[f"layer_{i}_{key}"]) for key in ("w", "b")} for i in range(4))
            norm = {key: jnp.asarray(a[key]) for key in ("mean", "std")}
            meta = json.loads(str(a["metadata"]))
        return cls(params, norm, ReactorConfig(**meta["config"])), meta


def train(data_dir, output_dir, max_epochs=200, seed=1234, patience=20):
    if not 1 <= max_epochs <= 200:
        raise ValueError("training requires 1 to 200 epochs")
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data_dir / "dataset.json").read_text())
    config = ReactorConfig(**meta["config"])
    fields = np.load(data_dir / "fields.npy", mmap_mode="r")
    controls, gusts = np.load(data_dir / "controls.npy"), np.load(data_dir / "disturbances.npy")
    train_ids, val_ids = np.array(meta["train_episodes"]), np.array(meta["validation_episodes"])
    # Whole-episode split; statistics use only training episodes and channels.
    sums, squares, count = np.zeros(2), np.zeros(2), 0
    for i in train_ids:
        block = np.asarray(fields[i], dtype=np.float64)
        sums += block.sum(axis=(0, 2, 3)); squares += (block**2).sum(axis=(0, 2, 3))
        count += block.shape[0] * config.nx * config.ny
    means = sums / count
    stds = np.maximum(np.sqrt(squares / count - means**2), [1., .01])
    norm = {"mean": jnp.asarray(means, dtype=jnp.float32), "std": jnp.asarray(stds, dtype=jnp.float32)}
    params = init_params(seed)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    rng = np.random.default_rng(seed)
    steps = controls.shape[1]
    pairs = np.array([(i, t) for i in train_ids for t in range(steps - 4)])
    # A fixed held-out rollout window, never shuffled into training.
    vi = val_ids[:16]
    vx = jnp.asarray(fields[vi, 0])
    vu = jnp.asarray(controls[vi, :20].swapaxes(0, 1))
    vg = jnp.asarray(gusts[vi, :20].swapaxes(0, 1))
    vy = jnp.asarray(fields[vi, 1:21].swapaxes(0, 1))

    @jax.jit
    def loss(p, x, u, g, target):
        predicted = surrogate_rollout(p, norm, x, u, g, config, False)
        errors = (predicted - target) / norm["std"][None, None, :, None, None]
        # One-step accuracy plus accumulated five-step prediction error.
        field_loss = jnp.mean(errors[0]**2) + jnp.mean(errors**2)
        outlet = jnp.mean(errors[..., 1, :, -1], axis=-1)
        return field_loss + 2 * jnp.mean(outlet**2)

    @jax.jit
    def update(p, o, x, u, g, target):
        value, grads = jax.value_and_grad(loss)(p, x, u, g, target)
        changes, o = optimizer.update(grads, o, p)
        return optax.apply_updates(p, changes), o, value

    @jax.jit
    def validate(p):
        predicted = surrogate_rollout(p, norm, vx, vu, vg, config, False)
        return jnp.mean(((predicted - vy) / norm["std"][None, None, :, None, None])**2)

    best, stale, history = np.inf, 0, []
    start = time.perf_counter()
    for epoch in range(max_epochs):
        rng.shuffle(pairs)
        losses = []
        for off in range(0, len(pairs) - 15, 16):
            ij = pairs[off:off + 16]; ids, ts = ij[:, 0], ij[:, 1]
            x = np.asarray(fields[ids, ts])
            u = np.stack([controls[ids, ts + j] for j in range(5)])
            g = np.stack([gusts[ids, ts + j] for j in range(5)])
            target = np.stack([fields[ids, ts + j + 1] for j in range(5)])
            params, opt_state, value = update(params, opt_state, x, u, g, target)
            losses.append(float(value))
        val = float(validate(params))
        record = {"epoch": epoch + 1, "training_loss": float(np.mean(losses)),
                  "validation_rollout_loss": val, "elapsed_s": time.perf_counter() - start}
        history.append(record)
        if np.isfinite(val) and val < best:
            best, stale = val, 0
            Surrogate(params, norm, config).save(output_dir / "surrogate.npz", {
                "seed": seed, "epoch": epoch + 1, "validation_loss": val,
                "inference_fraction_projection": "clip to [0, 1] after every predicted transition; training residual unrestricted",
                "architecture": "residual CNN, widths 6/16/16/16/2, dilations 1/2/4/1",
                "dataset_seed": meta["seed"], "status": "awaiting held-out test"})
        else:
            stale += 1
        (output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
        if stale >= patience:
            break
    (output_dir / "dataset.json").write_text(json.dumps(meta, indent=2))
    run = {"completed_epochs": len(history), "requested_max_epochs": max_epochs,
           "stopping_reason": "validation patience" if stale >= patience else "requested epoch budget",
           "checkpoint_sha256": hashlib.sha256((output_dir / "surrogate.npz").read_bytes()).hexdigest(),
           "inference_fraction_projection": "[0,1]; temperature unrestricted",
           "validation_selection_episodes": vi.tolist(), "platform": platform.platform(),
           "versions": {"jax": jax.__version__, "optax": optax.__version__, "numpy": np.__version__}}
    (output_dir / "training_run.json").write_text(json.dumps(run, indent=2))
    return history


def peak_underprediction(truth, prediction):
    """Worst underprediction of the spatial maximum at matching times."""
    true_peaks = np.asarray(truth)[..., 0, :, :].max(axis=(-2, -1))
    predicted_peaks = np.asarray(prediction)[..., 0, :, :].max(axis=(-2, -1))
    return float(np.maximum(true_peaks - predicted_peaks, 0).max())


def validate_surrogate(model, data_dir):
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "dataset.json").read_text())
    fields = np.load(data_dir / "fields.npy", mmap_mode="r")
    u, g = np.load(data_dir / "controls.npy"), np.load(data_dir / "disturbances.npy")
    rows = []
    for episode in meta["test_episodes"]:
        for start in (0, 20, 40):
            if start + 20 > u.shape[1]:
                continue
            truth = np.asarray(fields[episode, start + 1:start + 21])
            pred = np.asarray(model.rollout(fields[episode, start], u[episode, start:start + 20],
                                           g[episode, start:start + 20]))
            raw = np.asarray(surrogate_rollout(model.params, model.norm, fields[episode, start],
                         u[episode, start:start + 20], g[episode, start:start + 20], model.config, False))
            rows.append({"raw_fraction_bound_excess": float(np.maximum(-raw[:, 1], raw[:, 1] - 1).clip(0).max()),
                         "raw_temperature_rmse_K": float(np.sqrt(np.mean((truth[:, 0] - raw[:, 0])**2))),
                         "episode": episode, "start_step": start,
                         "temperature_rmse_K": float(np.sqrt(np.mean((truth[:, 0] - pred[:, 0])**2))),
                         "outlet_conversion_mae": float(np.mean(np.abs((truth[:, 1, :, -1] - pred[:, 1, :, -1]).mean(axis=-1)))),
                         "peak_underprediction_K": peak_underprediction(truth, pred),
                         "minimum_predicted_fraction": float(pred[:, 1].min()),
                         "maximum_predicted_fraction": float(pred[:, 1].max())})
    means = {key: float(np.mean([r[key] for r in rows])) for key in
             ("temperature_rmse_K", "outlet_conversion_mae", "peak_underprediction_K")}
    means["raw_fraction_bound_excess_max"] = max(r["raw_fraction_bound_excess"] for r in rows)
    means["inference_fraction_projection"] = "[0, 1]; temperature is not projected"
    means["passed"] = bool(means["temperature_rmse_K"] < 10 and means["outlet_conversion_mae"] < .02)
    means["peak_underprediction_p95_K"] = float(np.percentile([r["peak_underprediction_K"] for r in rows], 95))
    return {"summary": means, "windows": rows, "test_episodes": meta["test_episodes"]}
