"""Optional Apple-GPU inference for the exact JAX-trained residual CNN.

Training and reference physics remain JAX. This adapter changes only inference
execution; its field predictions and costs are checked against JAX in tests.
Install the optional ``thermal-metal`` dependency group on Apple silicon.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from thermal_reactor import heater_masks


class MetalScorer:
    def __init__(self, model, planner):
        cfg = model.config
        self.cfg, self.planner = cfg, planner
        # JAX HWIO -> MLX OHWI, with unchanged numeric values.
        layers = [(mx.array(np.asarray(p["w"]).transpose(3, 0, 1, 2)), mx.array(np.asarray(p["b"])))
                  for p in model.params]
        mean = mx.array(np.asarray(model.norm["mean"]))[None, None, None, :]
        std = mx.array(np.asarray(model.norm["std"]))[None, None, None, :]
        masks = mx.array(np.asarray(heater_masks(cfg)))
        yy, xx = np.meshgrid(np.linspace(-1, 1, cfg.ny), np.linspace(-1, 1, cfg.nx), indexing="ij")
        coordinates = mx.array(np.stack([xx, yy], -1).astype(np.float32))[None]
        y = (mx.arange(cfg.ny) + .5) / cfg.ny
        inlet_shape = cfg.inlet_temperature + cfg.transverse_amplitude * mx.cos(mx.pi * y)

        @mx.compile
        def step(fields, controls, gust):
            x = mx.transpose(fields, (0, 2, 3, 1))
            h = mx.einsum("bj,jyx->byx", controls, masks)[..., None]
            inlet = (inlet_shape[None, :, None, None] + gust[:, None, None, None] - mean[..., :1]) / std[..., :1]
            inlet = mx.broadcast_to(inlet, x.shape[:-1] + (1,))
            coords = mx.broadcast_to(coordinates, x.shape[:-1] + (2,))
            z = mx.concatenate([(x - mean) / std, h, inlet, coords], axis=-1)
            for i, ((w, b), dilation) in enumerate(zip(layers, (1, 2, 4, 1))):
                z = mx.pad(z, [(0, 0), (dilation, dilation), (dilation, dilation), (0, 0)], mode="edge")
                z = mx.conv2d(z, w, dilation=dilation) + b
                if i < 3:
                    z = z * mx.sigmoid(z)
            out = x + z * std
            out = mx.concatenate([out[..., :1], mx.clip(out[..., 1:], 0., 1.)], axis=-1)
            return mx.transpose(out, (0, 3, 1, 2))

        self.step = step

        def scores(initial, tapes, forecasts, previous):
            K, H, _ = tapes.shape; L = forecasts.shape[1]
            x = mx.broadcast_to(initial, (K * L,) + initial.shape)
            old_u = mx.broadcast_to(previous, (K * L, 4))
            accumulated = mx.zeros((K * L,))
            for t in range(H):
                ctrl = mx.repeat(tapes[:, t], L, axis=0)
                gust = mx.tile(forecasts[t], (K,))
                x = step(x, ctrl, gust)
                peak = mx.max(x[:, 0], axis=(-2, -1))
                conversion = 1 - mx.mean(x[:, 1, :, -1], axis=-1)
                cost = (mx.mean(ctrl, axis=-1)
                        + planner.quality_weight * (mx.maximum(cfg.conversion_target - conversion, 0) / .05)**2
                        + planner.heat_weight * (mx.maximum(peak - cfg.temperature_limit, 0) / 20)**2
                        + planner.slew_weight * mx.mean(((ctrl - old_u) / .1)**2, axis=-1))
                bad = (~mx.all(mx.isfinite(x), axis=(1, 2, 3))
                       | (mx.min(x[:, 0], axis=(-2, -1)) < 300)
                       | (mx.max(x[:, 0], axis=(-2, -1)) > 1600)
                       | (mx.min(x[:, 1], axis=(-2, -1)) < -.05)
                       | (mx.max(x[:, 1], axis=(-2, -1)) > 1.05))
                accumulated = accumulated + mx.where(bad, mx.inf, cost) / H
                old_u = ctrl
                # Evaluate each recurrent step; fusing all 20 reductions can
                # exceed Metal's argument-buffer limit and retain old fields.
                mx.eval(x, accumulated)
            return mx.mean(accumulated.reshape(K, L), axis=-1)
        self._scores = scores

    def score(self, initial, tapes, forecasts, previous):
        args = [mx.array(np.asarray(x, dtype=np.float32)) for x in (initial, tapes, forecasts, previous)]
        return np.asarray(self._scores(*args))

    def predict(self, fields, controls, gust):
        return np.asarray(self.step(*[mx.array(np.asarray(x, dtype=np.float32)) for x in (fields, controls, gust)]))
