"""Small Gaussian importance-sampling primitives used by the MPPI examples.

The reference distribution stays fixed when the sampling mean changes. Costs
exclude the Gaussian reference penalty: ``log_ratio`` supplies it exactly once.
"""
from __future__ import annotations

import numpy as np


def normalized_weights(costs, temperature, log_ratio=None):
    """Return stable normalized weights and effective sample size.

    ``log_ratio`` is log(reference density / sampling density). Nonfinite
    candidates receive zero weight; an entirely invalid batch raises ValueError.
    Shifting the finite costs before division avoids overflow for large offsets.
    """
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 1 or not costs.size:
        raise ValueError("costs must be a nonempty vector")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    ratio = np.zeros_like(costs) if log_ratio is None else np.asarray(log_ratio, dtype=float)
    if ratio.shape != costs.shape:
        raise ValueError("log_ratio must have the same shape as costs")
    valid = np.isfinite(costs) & np.isfinite(ratio)
    if not valid.any():
        raise ValueError("all candidate weights are invalid")
    logits = np.full_like(costs, -np.inf)
    # Wider arithmetic where available; on platforms with 64-bit long double,
    # enormous positive gaps can safely saturate to zero probability.
    with np.errstate(over="ignore", under="ignore"):
        shifted = costs[valid].astype(np.longdouble) - np.min(costs[valid]).astype(np.longdouble)
        score = -shifted / temperature + ratio[valid].astype(np.longdouble)
        score -= np.max(score)
        logits[valid] = np.asarray(score, dtype=float)
    weights = np.exp(logits)
    weights /= weights.sum()
    return weights, float(1.0 / np.dot(weights, weights))


def gaussian_log_ratio(samples, mean, variance, reference_mean=0.0):
    """Log p/q for diagonal Gaussians with equal, fixed covariance.

    Samples have shape (batch, ...); all remaining axes are trajectory-parameter
    axes. ``variance`` broadcasts over those axes and is strictly positive.
    """
    samples = np.asarray(samples, dtype=float)
    mean, variance = np.asarray(mean, dtype=float), np.asarray(variance, dtype=float)
    reference_mean = np.asarray(reference_mean, dtype=float)
    if samples.ndim < 2:
        raise ValueError("samples must have a batch and a parameter axis")
    if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError("variance must be finite and positive")
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(reference_mean)):
        raise ValueError("Gaussian means must be finite")
    # Difference of squares evaluated in expanded form avoids cancellation.
    terms = ((reference_mean - mean) * samples + 0.5 * (mean**2 - reference_mean**2)) / variance
    return np.sum(terms, axis=tuple(range(1, samples.ndim)))


def gaussian_mppi_update(samples, costs, mean, variance, temperature, reference_mean=0.0):
    """Project the tilted reference law onto a Gaussian with fixed covariance.

    Returns (updated mean, weights, ESS). A failed batch raises ValueError so a
    caller can retain its previous feasible plan explicitly.
    """
    samples = np.asarray(samples, dtype=float)
    costs = np.asarray(costs, dtype=float).copy()
    invalid = ~np.all(np.isfinite(samples), axis=tuple(range(1, samples.ndim)))
    costs[invalid] = np.inf
    ratio = gaussian_log_ratio(samples, mean, variance, reference_mean)
    weights, ess = normalized_weights(costs, temperature, ratio)
    safe_samples = np.where(np.isfinite(samples), samples, 0.0)
    return np.tensordot(weights, safe_samples, axes=1), weights, ess
