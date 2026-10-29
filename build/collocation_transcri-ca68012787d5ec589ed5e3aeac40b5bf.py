"""Construct polynomial collocation rules and evaluate their finite NLP.

The routines use explicit polynomial coefficients for teaching and modest
degrees. They do not select a mesh, scale variables, or choose an NLP solver.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Polynomial


def cardinal_polynomials(nodes):
    """Return the Lagrange basis for distinct nodes in [0, 1]."""
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 1 or nodes.size == 0:
        raise ValueError("nodes must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(nodes)) or np.any((nodes < 0) | (nodes > 1)):
        raise ValueError("nodes must be finite and lie in [0, 1]")
    if np.unique(nodes).size != nodes.size:
        raise ValueError("nodes must be distinct")
    basis = []
    for j, node in enumerate(nodes):
        polynomial = Polynomial([1.0])
        for m, other in enumerate(nodes):
            if m != j:
                polynomial *= Polynomial([-other, 1.0]) / (node - other)
        basis.append(polynomial)
    return basis


@dataclass
class CollocationRule:
    nodes: np.ndarray
    control_nodes: np.ndarray
    A: np.ndarray
    b: np.ndarray
    B: np.ndarray
    control_left: np.ndarray
    control_right: np.ndarray


def make_rule(nodes, control_nodes):
    """Precompute integration weights and control evaluation weights.

    A[i,j] = integral from 0 to nodes[i] of slope basis j.
    b[j] = integral from 0 to 1 of slope basis j.
    B[j,r] = control basis r evaluated at collocation node j.
    """
    slope_basis = cardinal_polynomials(nodes)
    control_basis = cardinal_polynomials(control_nodes)
    nodes = np.asarray(nodes, dtype=float)
    antiderivatives = [polynomial.integ() for polynomial in slope_basis]
    A = np.column_stack([p(nodes) - p(0.0) for p in antiderivatives])
    b = np.array([p(1.0) - p(0.0) for p in antiderivatives])
    B = np.column_stack([p(nodes) for p in control_basis])
    return CollocationRule(
        nodes=nodes,
        control_nodes=np.asarray(control_nodes, dtype=float),
        A=A,
        b=b,
        B=B,
        control_left=np.array([p(0.0) for p in control_basis]),
        control_right=np.array([p(1.0) for p in control_basis]),
    )


def transcribed_problem(
    rule, mesh, x_mesh, x_stage, u_support,
    dynamics, running_cost, terminal_cost, boundary, path=None,
    *, continuous_control=False,
):
    """Return objective F, inequalities G <= 0, and equalities H = 0.

    x_mesh: (N+1, n); x_stage: (N, s, n); u_support: (N, d_u+1, m).
    dynamics(x,u,t) returns n slopes; running_cost(x,u,t) returns a scalar.
    terminal_cost(x_final,t_final) returns a scalar.
    boundary(x_initial,x_final,t_final) returns equality residuals, including
    the prescribed initial condition. path(x,u,t), if supplied, returns
    inequality residuals sampled at the collocation stages only.
    """
    mesh = np.asarray(mesh, dtype=float)
    x_mesh, x_stage, u_support = map(
        lambda a: np.asarray(a, dtype=float), (x_mesh, x_stage, u_support)
    )
    if mesh.ndim != 1 or len(mesh) < 2 or not np.all(np.isfinite(mesh)):
        raise ValueError("mesh must be a finite one-dimensional array")
    widths = np.diff(mesh)
    if np.any(widths <= 0):
        raise ValueError("mesh times must be strictly increasing")
    N, s, r = len(widths), len(rule.nodes), len(rule.control_nodes)
    if x_mesh.ndim != 2 or x_mesh.shape[0] != N + 1:
        raise ValueError("x_mesh must have shape (N+1, n)")
    if x_stage.shape != (N, s, x_mesh.shape[1]):
        raise ValueError("x_stage must have shape (N, s, n)")
    if u_support.ndim != 3 or u_support.shape[:2] != (N, r):
        raise ValueError("u_support must have shape (N, d_u+1, m)")

    objective = float(terminal_cost(x_mesh[-1], mesh[-1]))
    equalities = [np.asarray(boundary(x_mesh[0], x_mesh[-1], mesh[-1])).ravel()]
    inequalities = []
    for k, width in enumerate(widths):
        stage_times = mesh[k] + width * rule.nodes
        stage_controls = rule.B @ u_support[k]
        slopes = np.stack([
            dynamics(x, u, t)
            for x, u, t in zip(x_stage[k], stage_controls, stage_times)
        ])
        rates = np.array([
            running_cost(x, u, t)
            for x, u, t in zip(x_stage[k], stage_controls, stage_times)
        ])
        equalities.append((x_stage[k] - x_mesh[k] - width * rule.A @ slopes).ravel())
        equalities.append(x_mesh[k + 1] - x_mesh[k] - width * rule.b @ slopes)
        objective += width * rule.b @ rates
        if path is not None:
            inequalities.extend(
                np.asarray(path(x, u, t)).ravel()
                for x, u, t in zip(x_stage[k], stage_controls, stage_times)
            )
        if continuous_control and k + 1 < N:
            equalities.append(
                rule.control_right @ u_support[k]
                - rule.control_left @ u_support[k + 1]
            )
    G = np.concatenate(inequalities) if inequalities else np.empty(0)
    return float(objective), G, np.concatenate(equalities)
