r"""Small policy-evaluation problem used by the test-function widget.

State s in [0, 1], fixed policy with dynamics s' = 0.8 sqrt(s) + 0.2 u, where
u takes the values 0, 0.5, 1 with probabilities 1/4, 1/2, 1/4. The reward is a
smoothed step r(s) = 1 / (1 + exp(-(s - 0.7) / 0.03)) and gamma = 0.9.

Three weighted-residual solutions are computed with the same basis (hat
functions on uniform nodes, or shifted Legendre polynomials with Chebyshev
collocation nodes): collocation, Galerkin and least squares. Each one is then
tested against the three families of test functions. The widget
interactive/test-functions.html implements the same computation in the browser.
"""
import numpy as np

GAMMA = 0.9
SHOCKS = np.array([0.0, 0.5, 1.0])
PROBS = np.array([0.25, 0.5, 0.25])
GRID = np.linspace(0.0, 1.0, 801)
QW = np.full(GRID.size, GRID[1] - GRID[0])
QW[[0, -1]] *= 0.5


def reward(s):
    return 1.0 / (1.0 + np.exp(-(s - 0.7) / 0.03))


def successors(s):
    return 0.8 * np.sqrt(np.asarray(s))[..., None] + 0.2 * SHOCKS


def hat_nodes(n):
    return np.linspace(0.0, 1.0, n)


def chebyshev_nodes(n):
    k = np.arange(1, n + 1)
    return np.sort(0.5 - 0.5 * np.cos((2 * k - 1) * np.pi / (2 * n)))


def hats(s, n):
    nodes = hat_nodes(n)
    if n == 1:
        return np.ones(np.shape(s) + (1,))
    h = nodes[1] - nodes[0]
    return np.clip(1.0 - np.abs(np.asarray(s)[..., None] - nodes) / h, 0.0, None)


def legendre(s, n):
    x = 2.0 * np.asarray(s) - 1.0
    out = np.empty(np.shape(s) + (n,))
    out[..., 0] = 1.0
    if n > 1:
        out[..., 1] = x
    for k in range(2, n):
        out[..., k] = ((2 * k - 1) * x * out[..., k - 1] - (k - 1) * out[..., k - 2]) / k
    return out


def basis(kind, n):
    phi = hats if kind == "hat" else legendre
    nodes = hat_nodes(n) if kind == "hat" else chebyshev_nodes(n)
    return phi, nodes


def exact_value(iterations=400):
    """Value of the policy on GRID, by successive approximation with linear interpolation."""
    v = np.zeros(GRID.size)
    nxt = successors(GRID)
    for _ in range(iterations):
        v = reward(GRID) + GAMMA * (np.interp(nxt, GRID, v) @ PROBS)
    return v


def residual_parts(phi, n, s):
    """R(s; theta) = b(s) + D(s) theta with b = r and D = gamma P phi - phi."""
    D = GAMMA * np.einsum("...uj,u->...j", phi(successors(s), n), PROBS) - phi(s, n)
    return reward(s), D


def solve(kind, n, method):
    phi, nodes = basis(kind, n)
    if method == "collocation":
        b, D = residual_parts(phi, n, nodes)
        return np.linalg.solve(D, -b)
    b, D = residual_parts(phi, n, GRID)
    test = phi(GRID, n) if method == "galerkin" else D
    A = test.T @ (QW[:, None] * D)
    return np.linalg.solve(A, -(test.T @ (QW * b)))


def test_values(kind, n, theta):
    """Inner products of the residual with each family of unit-norm test functions."""
    phi, nodes = basis(kind, n)
    b, D = residual_parts(phi, n, nodes)
    out = {"collocation": b + D @ theta}
    b, D = residual_parts(phi, n, GRID)
    R = b + D @ theta
    for name, P in (("galerkin", phi(GRID, n)), ("least squares", D)):
        norms = np.sqrt(QW @ P**2)
        out[name] = (P.T @ (QW * R)) / norms
    return out, R


def table(kind="hat", n=6):
    """Largest |test value| for each (solution, test family) pair, and the value error."""
    v = exact_value()
    rows = {}
    for method in ("collocation", "galerkin", "least squares"):
        theta = solve(kind, n, method)
        tests, R = test_values(kind, n, theta)
        phi, _ = basis(kind, n)
        err = np.max(np.abs(phi(GRID, n) @ theta - v))
        rows[method] = {k: float(np.max(np.abs(t))) for k, t in tests.items()}
        rows[method]["residual L2"] = float(np.sqrt(QW @ R**2))
        rows[method]["value error (sup)"] = float(err)
    return rows
