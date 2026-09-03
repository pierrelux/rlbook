from __future__ import annotations

import numpy as np


NODES = np.array([0.0, 0.5, 1.0])
D = np.array(
    [
        [-3.0, 4.0, -1.0],
        [-1.0, 0.0, 1.0],
        [1.0, -4.0, 3.0],
    ]
)
WEIGHTS = np.array([1.0, 4.0, 1.0]) / 6.0


def _cardinal_values(points: np.ndarray) -> np.ndarray:
    values = np.ones((points.size, NODES.size))
    for j, node in enumerate(NODES):
        for m, other_node in enumerate(NODES):
            if m != j:
                values[:, j] *= (points - other_node) / (node - other_node)
    return values


def test_three_node_cardinal_identity() -> None:
    np.testing.assert_allclose(_cardinal_values(NODES), np.eye(3), atol=1e-14)


def test_three_node_differentiation_is_exact_for_quadratics() -> None:
    coefficients = np.array([1.3, -0.7, 2.1])
    nodal_values = (
        coefficients[0]
        + coefficients[1] * NODES
        + coefficients[2] * NODES**2
    )
    exact_derivatives = coefficients[1] + 2.0 * coefficients[2] * NODES

    np.testing.assert_allclose(D @ nodal_values, exact_derivatives, atol=1e-14)


def test_three_node_quadrature_has_simpson_weights() -> None:
    np.testing.assert_allclose(WEIGHTS, [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0])

    coefficients = np.array([1.3, -0.7, 2.1])
    nodal_values = (
        coefficients[0]
        + coefficients[1] * NODES
        + coefficients[2] * NODES**2
    )
    exact_integral = (
        coefficients[0] + coefficients[1] / 2.0 + coefficients[2] / 3.0
    )

    np.testing.assert_allclose(WEIGHTS @ nodal_values, exact_integral, atol=1e-14)
