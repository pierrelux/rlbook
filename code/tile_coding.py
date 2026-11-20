r"""Tile coding as a linear architecture: its span, its kernel, and where it sits
between averagers and Galerkin projections.

A tiling of [0, 1]^d with tile width h and offset o is the partition into the
cells floor((s + o) / h). Tile coding stacks m tilings (K in the text) with different offsets;
a state s turns on one tile per tiling, so its feature vector phi(s) is binary
with exactly m ones and v(s; w) = phi(s)^T w is the sum of m weights.

Quantities computed here, all on finite grids of states:

* the shared-tile kernel k(s, s') = phi(s)^T phi(s') / m, for uniform offsets
  (a staircase) and for random offsets (whose mean is the hat
  max(0, 1 - |s - s'| / h), Rahimi and Recht's random binning);
* the least-squares (Galerkin) projection onto the span of the tile features,
  its smallest entry and its sup-norm operator norm (max absolute row sum);
* the average of the m single-tiling aggregation projections, an averager;
* the inner products of hashed tile features, whose bias from collisions is
  (m^2 - shared) / M for a table of M weights and an unsigned hash.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def tile_indices(S, h, offsets):
    """Index of the active tile of each tiling for each state.

    S: (n_states, d) array in [0, 1)^d. offsets: (m, d) array in [0, h).
    Returns an (n_states, m) integer array; tiles of different tilings get
    different indices.
    """
    S = np.atleast_2d(S)
    n_states, d = S.shape
    per_axis = int(np.floor((1.0 + h) / h)) + 1
    cols = []
    for k, o in enumerate(np.atleast_2d(offsets)):
        cell = np.floor((S + o) / h).astype(int)
        flat = np.ravel_multi_index(cell.T, (per_axis,) * d)
        cols.append(k * per_axis ** d + flat)
    return np.stack(cols, axis=1)


def feature_matrix(idx):
    """Binary feature matrix from tile indices, dropping tiles no state uses."""
    n_states, m = idx.shape
    used, inv = np.unique(idx, return_inverse=True)
    Phi = np.zeros((n_states, len(used)))
    np.add.at(Phi, (np.repeat(np.arange(n_states), m), inv.ravel()), 1.0)
    return Phi


def uniform_offsets(m, h, d=1, displacement=None):
    """Offsets k * (h / m) * displacement (mod h), k = 0..m-1.

    displacement = (1, 1, ..., 1) is the diagonal default; Sutton and Barto
    recommend (1, 3, 5, ...)."""
    if displacement is None:
        displacement = np.ones(d)
    k = np.arange(m)[:, None]
    return (k * np.asarray(displacement, float)[None, :] * h / m) % h


def random_offsets(m, h, d=1, rng=None):
    rng = np.random.default_rng(rng)
    return rng.uniform(0.0, h, size=(m, d))


def grid(n, d=1):
    """Cell centers of an n^d grid on [0, 1)^d."""
    g = (np.arange(n) + 0.5) / n
    mesh = np.meshgrid(*([g] * d), indexing="ij")
    return np.stack([x.ravel() for x in mesh], axis=1)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

def shared_fraction(s0, S, h, offsets):
    """k(s0, s) = (number of tiles shared by s0 and s) / m for each s in S."""
    i0 = tile_indices(np.atleast_2d(s0), h, offsets)[0]
    idx = tile_indices(S, h, offsets)
    return (idx == i0[None, :]).mean(axis=1)


def hat(delta, h):
    return np.prod(np.clip(1.0 - np.abs(np.atleast_2d(delta)) / h, 0.0, None), axis=1)


def kernel_report_1d(m=8, h=0.25, s0=0.5, n=801, trials=4000, seed=0):
    """Staircase for uniform offsets, hat on average for random offsets."""
    S = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    target = hat(S - s0, h)
    k_uniform = shared_fraction([s0], S, h, uniform_offsets(m, h))
    rng = np.random.default_rng(seed)
    k_random = np.zeros(n)
    for _ in range(trials):
        k_random += shared_fraction([s0], S, h, random_offsets(m, h, rng=rng))
    k_random /= trials
    return {
        "uniform: max |k - hat|": np.abs(k_uniform - target).max(),
        "random (mean of %d draws): max |k - hat|" % trials: np.abs(k_random - target).max(),
        "1/K": 1.0 / m,
    }


def kernel_2d_anisotropy(m=8, h=0.25, r=0.1):
    """Shared fraction at displacement r along the diagonal and the
    antidiagonal, for diagonal and (1, 3) displacement vectors."""
    s0 = np.array([0.5, 0.5])
    out = {}
    for name, disp in (("diagonal (1, 1)", (1, 1)), ("asymmetric (1, 3)", (1, 3))):
        off = uniform_offsets(m, h, 2, disp)
        along = shared_fraction(s0, (s0 + r / np.sqrt(2) * np.array([1, 1]))[None], h, off)[0]
        across = shared_fraction(s0, (s0 + r / np.sqrt(2) * np.array([1, -1]))[None], h, off)[0]
        out[name] = (along, across)
    return out


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------

def galerkin_projection(Phi, xi):
    """xi-orthogonal projection onto span(Phi); pinv because the m tilings
    each sum to the constant function, so Phi has dependent columns."""
    G = Phi.T @ (xi[:, None] * Phi)
    return Phi @ np.linalg.pinv(G) @ (Phi.T * xi)


def averaged_aggregation(idx, xi):
    """(1/m) sum_k Proj_k, Proj_k the aggregation onto tiling k alone."""
    n_states, m = idx.shape
    P = np.zeros((n_states, n_states))
    for k in range(m):
        Phi_k = feature_matrix(idx[:, [k]])
        P += galerkin_projection(Phi_k, xi) / m
    return P


def projection_report(m=4, h=0.25, n=32, d=1, displacement=None):
    S = grid(n, d)
    xi = np.full(len(S), 1.0 / len(S))
    idx = tile_indices(S, h, uniform_offsets(m, h, d, displacement))
    Phi = feature_matrix(idx)
    P_joint = galerkin_projection(Phi, xi)
    P_avg = averaged_aggregation(idx, xi)
    fine_cells = len(np.unique(idx, axis=0))
    return {
        "states": len(S),
        "weights": Phi.shape[1],
        "rank": np.linalg.matrix_rank(Phi),
        "fine cells": fine_cells,
        "joint LS: min entry": P_joint.min(),
        "joint LS: sup norm": np.abs(P_joint).sum(axis=1).max(),
        "averaged: min entry": P_avg.min(),
        "averaged: sup norm": np.abs(P_avg).sum(axis=1).max(),
        "averaged: |P^2 - P|": np.abs(P_avg @ P_avg - P_avg).max(),
    }


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def hashing_report(m=8, h=0.25, M=64, d=2, pairs=400, hashes=400, seed=0):
    """Mean inner product of hashed features over random hash functions,
    against shared + (m^2 - shared) / M (unsigned) and shared (signed)."""
    rng = np.random.default_rng(seed)
    off = uniform_offsets(m, h, d, np.arange(1, 2 * d, 2))
    A = rng.uniform(0, 1, size=(pairs, d))
    B = np.clip(A + rng.normal(0, h / 2, size=(pairs, d)), 0, 0.999)
    ia, ib = tile_indices(A, h, off), tile_indices(B, h, off)
    shared = (ia == ib).sum(axis=1).astype(float)
    n_tiles = ia.max() + ib.max() + 2
    unsigned = np.zeros(pairs)
    signed = np.zeros(pairs)
    for _ in range(hashes):
        bucket = rng.integers(0, M, size=n_tiles)
        sign = rng.choice([-1.0, 1.0], size=n_tiles)
        for j in range(pairs):
            ha, hb = bucket[ia[j]], bucket[ib[j]]
            same = ha[:, None] == hb[None, :]
            unsigned[j] += same.sum()
            signed[j] += (same * sign[ia[j]][:, None] * sign[ib[j]][None, :]).sum()
    unsigned /= hashes
    signed /= hashes
    predicted = shared + (m * m - shared) / M
    return {
        "mean shared tiles": shared.mean(),
        "unsigned hash: mean inner product": unsigned.mean(),
        "predicted shared + (K^2 - shared)/M": predicted.mean(),
        "signed hash: mean inner product": signed.mean(),
    }


def smoother_matches_kernel(m=8, h=0.25, n=400):
    """Interior rows of the averaged aggregation are k(s, .) / sum k(s, .)."""
    S = grid(n, 1)
    xi = np.full(n, 1.0 / n)
    off = uniform_offsets(m, h)
    idx = tile_indices(S, h, off)
    P = averaged_aggregation(idx, xi)
    K = (idx[:, None, :] == idx[None, :, :]).mean(axis=2)
    interior = (S[:, 0] > h) & (S[:, 0] < 1 - h)
    W = K[interior] / K[interior].sum(axis=1, keepdims=True)
    return np.abs(P[interior] - W).max()


# ---------------------------------------------------------------------------
# Learning from samples (the second tab of the widget)
# ---------------------------------------------------------------------------

def target(s):
    return np.sin(2 * np.pi * s) + 0.8 * (s > 0.6)


def learning_curves(m=8, h=0.2, alpha=0.1, sigma=0.3, checkpoints=(30, 100, 300, 1000, 3000),
                    seeds=20):
    """RMSE of a coarse tiling (width h), a fine tiling (width h/m) and m
    tilings of width h, all trained by w <- w + (alpha/m') phi (y - phi^T w)
    on the same noisy samples, averaged over seeds."""
    learners = {"coarse, 1 tiling of width h": (1, h),
                "fine, 1 tiling of width h/K": (1, h / m),
                "tile coding, K tilings of width h": (m, h)}
    G = grid(400)
    out = {}
    for name, (mm, hh) in learners.items():
        off = uniform_offsets(mm, hh)
        iG = tile_indices(G, hh, off)
        curves = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            T = max(checkpoints)
            s = rng.uniform(0, 1, size=(T, 1))
            y = target(s[:, 0]) + sigma * rng.normal(size=T)
            I = tile_indices(s, hh, off)
            w = np.zeros(max(iG.max(), I.max()) + 1)
            errs = []
            for t in range(T):
                i = I[t]
                w[i] += alpha / mm * (y[t] - w[i].sum())
                if t + 1 in checkpoints:
                    errs.append(np.sqrt(np.mean((w[iG].sum(axis=1) - target(G[:, 0])) ** 2)))
            curves.append(errs)
        out[name] = np.mean(curves, axis=0)
    return out
