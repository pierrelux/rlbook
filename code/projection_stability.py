r"""Projected Bellman equations on a finite state space: which projections keep
the iteration stable, and in which norm.

Every weighted-residual method of the parametric value function chapter
produces a fixed-point equation v = Proj L v, where Proj is a projection onto
the span of the features Phi (m x n) along the directions that the tests
Psi cannot see:

    Proj = Phi (Psi^T Xi Phi)^{-1} Psi^T Xi.

Collocation takes Psi^T Xi = E, the n x m matrix that selects the nodes.
Galerkin takes Psi = Phi. This module builds these matrices for a discretized
version of the policy-evaluation chain used by the test-function widget
(s' = 0.8 sqrt(s) + 0.2 u on [0, 1], with uniform restarts) and for Baird's star counterexample, and
reports the quantities that decide stability:

* the sup-norm operator norm of Proj (max absolute row sum), which is at most
  one exactly when Proj is an averager (Gordon's route);
* the xi-norm operator norm of Proj, for xi the stationary distribution of
  P_pi (the route of Tsitsiklis and Van Roy);
* the spectral radius of gamma Proj P_pi, which decides whether projected
  value iteration v <- Proj L_pi v converges;
* the smallest real part of an eigenvalue of A = Phi^T Xi (I - gamma P_pi) Phi,
  which decides whether expected TD(0), theta <- theta + alpha (b - A theta),
  converges for small step sizes.
"""
import numpy as np

import test_functions_demo as tf

GAMMA = tf.GAMMA


# ---------------------------------------------------------------------------
# Generic linear algebra
# ---------------------------------------------------------------------------

def oblique_projection(Phi, Psi, xi):
    """Proj = Phi (Psi^T Xi Phi)^{-1} Psi^T Xi."""
    T = Psi.T * xi
    return Phi @ np.linalg.solve(T @ Phi, T)


def collocation_projection(Phi, node_index):
    """Interpolation through the nodes: Proj = Phi (E Phi)^{-1} E."""
    E = np.zeros((len(node_index), Phi.shape[0]))
    E[np.arange(len(node_index)), node_index] = 1.0
    return Phi @ np.linalg.solve(E @ Phi, E)


def sup_norm(M):
    """Operator norm induced by the sup norm: largest absolute row sum."""
    return float(np.max(np.abs(M).sum(axis=1)))


def xi_norm(M, xi):
    """Operator norm induced by ||v||_xi = sqrt(sum xi v^2)."""
    r = np.sqrt(xi)
    return float(np.linalg.norm(r[:, None] * M / r[None, :], 2))


def spectral_radius(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def stationary(P, iters=20000):
    xi = np.full(P.shape[0], 1.0 / P.shape[0])
    for _ in range(iters):
        xi = xi @ P
    return xi / xi.sum()


def td_matrix(Phi, xi, P, gamma):
    """A = Phi^T Xi (I - gamma P) Phi, the matrix of expected TD(0)."""
    return Phi.T @ (xi[:, None] * (Phi - gamma * P @ Phi))


# ---------------------------------------------------------------------------
# Two-state counterexample of Tsitsiklis and Van Roy (1996)
# ---------------------------------------------------------------------------

def two_state(xi1, gamma):
    """Feature (1, 2), state 1 -> state 2 -> state 2, weights (xi1, 1 - xi1).

    Returns the factor of projected value iteration theta <- c theta and the
    scalar A of expected TD(0), theta <- theta + alpha (0 - A theta).
    """
    xi = np.array([xi1, 1.0 - xi1])
    Phi = np.array([[1.0], [2.0]])
    P = np.array([[0.0, 1.0], [0.0, 1.0]])
    Proj = oblique_projection(Phi, Phi, xi)
    factor = float((np.linalg.pinv(Phi) @ (gamma * Proj @ P @ Phi))[0, 0])
    A = float(td_matrix(Phi, xi, P, gamma)[0, 0])
    return factor, A


def two_state_td(xi1, gamma, alpha=0.05, steps=400, seed=0):
    """Sampled TD(0) on the two-state example, states drawn i.i.d. from xi."""
    rng = np.random.default_rng(seed)
    phi = np.array([1.0, 2.0])
    theta = 1.0
    for _ in range(steps):
        s = 0 if rng.random() < xi1 else 1
        delta = gamma * phi[1] * theta - phi[s] * theta  # both states move to state 2
        theta += alpha * phi[s] * delta
    return theta


# ---------------------------------------------------------------------------
# Discretized policy-evaluation chain of the test-function widget
# ---------------------------------------------------------------------------

def chain(m=201, restart=0.2):
    """Grid, transition matrix and reward of the widget's chain, with restarts.

    With probability 1 - restart the next state is 0.8 sqrt(s) + 0.2 u,
    spread over the two neighbouring grid points by linear interpolation; with
    probability restart it is drawn uniformly from the grid. Without restarts
    the chain never returns below s = 0.64 and its stationary distribution
    gives no weight to the low states.
    """
    grid = np.linspace(0.0, 1.0, m)
    nxt = tf.successors(grid)  # m x 3
    P = np.zeros((m, m))
    h = grid[1] - grid[0]
    for u, p in enumerate(tf.PROBS):
        x = np.clip(nxt[:, u], 0.0, 1.0)
        j = np.minimum((x / h).astype(int), m - 2)
        w = (x - grid[j]) / h
        P[np.arange(m), j] += p * (1.0 - w)
        P[np.arange(m), j + 1] += p * w
    P = (1.0 - restart) * P + restart / m
    return grid, P, tf.reward(grid)


def nearest(grid, points):
    return np.array([int(np.argmin(np.abs(grid - p))) for p in points])


def aggregation_features(grid, n):
    k = np.minimum((grid * n).astype(int), n - 1)
    Phi = np.zeros((grid.size, n))
    Phi[np.arange(grid.size), k] = 1.0
    return Phi


def chain_table(n=6, m=201, gamma=GAMMA):
    """Stability diagnostics of six projections on the same chain."""
    grid, P, r = chain(m)
    xi_stat = stationary(P)
    uniform = np.full(m, 1.0 / m)
    v_pi = np.linalg.solve(np.eye(m) - gamma * P, r)

    hat_nodes = nearest(grid, tf.hat_nodes(n))
    cheb_nodes = nearest(grid, tf.chebyshev_nodes(n))
    Hat = tf.hats(grid, n)
    Leg = tf.legendre(grid, n)
    Agg = aggregation_features(grid, n)

    cases = [
        ("collocation, hats", Hat, collocation_projection(Hat, hat_nodes), None),
        ("collocation, polynomial", Leg, collocation_projection(Leg, cheb_nodes), None),
        ("Galerkin, hats, uniform", Hat, oblique_projection(Hat, Hat, uniform), uniform),
        ("Galerkin, hats, stationary", Hat, oblique_projection(Hat, Hat, xi_stat), xi_stat),
        ("Galerkin, polynomial, stationary", Leg, oblique_projection(Leg, Leg, xi_stat), xi_stat),
        ("aggregation, uniform", Agg, oblique_projection(Agg, Agg, uniform), uniform),
    ]
    rows = []
    for name, Phi, Proj, w in cases:
        v = np.linalg.solve(np.eye(m) - gamma * Proj @ P, Proj @ r)
        row = {
            "method": name,
            "sup |P|": sup_norm(Proj),
            "xi |P|": xi_norm(Proj, xi_stat),
            "min entry": float(Proj.min()),
            "rho(gamma P L)": spectral_radius(gamma * Proj @ P),
            "min Re eig A": (float(np.min(np.linalg.eigvals(td_matrix(Phi, w, P, gamma)).real))
                             if w is not None else float("nan")),
            "sup error": float(np.max(np.abs(v - v_pi))),
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Baird's star counterexample (Baird 1995; Sutton and Barto 2018, Example 11.1)
# ---------------------------------------------------------------------------

def baird():
    """Seven states, eight weights, target policy always moves to state 7.

    The behaviour policy makes the next state uniform over the seven states,
    so the data weight the states uniformly. All rewards are zero.
    """
    Phi = np.zeros((7, 8))
    for i in range(6):
        Phi[i, i] = 2.0
        Phi[i, 7] = 1.0
    Phi[6, 6] = 1.0
    Phi[6, 7] = 2.0
    P_target = np.zeros((7, 7))
    P_target[:, 6] = 1.0
    P_behaviour = np.full((7, 7), 1.0 / 7.0)
    return Phi, P_target, P_behaviour


def baird_run(method, gamma=0.99, alpha=0.01, beta=0.05, steps=2000):
    """Expected updates on Baird's star. Returns the largest |v(s; theta)| along
    the iterations (the true value function is zero).

    td:       theta <- theta + alpha (b - A theta), A = Phi^T Xi (I - gamma P_pi) Phi
    tdc:      the TD with gradient correction of Sutton et al. (2009), expected form
    emphatic: emphatic TD(0) of Sutton, Mahmood and White (2016), expected form,
              A_F = Phi^T F (I - gamma P_pi) Phi with f = (I - gamma P_pi^T)^{-1} xi
    """
    Phi, P, Pb = baird()
    xi = stationary(Pb)
    theta = np.array([1, 1, 1, 1, 1, 1, 10, 1], dtype=float)
    u = np.zeros(8)
    C = Phi.T @ (xi[:, None] * Phi)
    A = td_matrix(Phi, xi, P, gamma)
    if method == "emphatic":
        f = np.linalg.solve(np.eye(7) - gamma * P.T, xi)
        A = Phi.T @ (f[:, None] * (Phi - gamma * P @ Phi))
    norms = [np.max(np.abs(Phi @ theta))]
    for _ in range(steps):
        if method in ("td", "emphatic"):
            theta = theta - alpha * A @ theta
        elif method == "tdc":
            delta = -A @ theta  # E[delta phi], rewards are zero
            corr = gamma * (P @ Phi).T @ (xi[:, None] * Phi) @ u  # E[phi(s') phi(s)^T] u
            theta = theta + alpha * (delta - corr)
            u = u + beta * (delta - C @ u)
        norms.append(np.max(np.abs(Phi @ theta)))
    return np.array(norms), A


def baird_table(gamma=0.99):
    Phi, P, Pb = baird()
    xi = stationary(Pb)
    out = {
        "rank of Phi": int(np.linalg.matrix_rank(Phi)),
        "xi-norm of gamma P_pi": xi_norm(gamma * P, xi),
    }
    for method in ("td", "tdc", "emphatic"):
        norms, A = baird_run(method, gamma)
        out[method] = {
            "min Re eig": float(np.min(np.linalg.eigvals(A).real)),
            "max |v| at 0, 500, 2000": (float(norms[0]), float(norms[500]), float(norms[-1])),
        }
    return out
