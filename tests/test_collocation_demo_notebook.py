"""Run the collocation demo notebook and check the numbers it teaches."""

import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "lab" / "notebooks" / "08_collocation_from_nodes.ipynb"
)


def code_cells():
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    return ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]


def run(*replacements, cells=None):
    """Execute the code cells, optionally after changing a line as a reader would."""
    source = "\n\n".join(code_cells()[:cells]).replace("plt.show()", "plt.close('all')")
    for old, new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    namespace = {}
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace


def cells_through(text):
    """Number of code cells up to and including the first one that contains text."""
    return next(i for i, cell in enumerate(code_cells()) if text in cell) + 1


@pytest.fixture(scope="module")
def demo():
    return run()


def test_rules_match_the_hand_calculations(demo):
    np.testing.assert_allclose(demo["A"], [[5/12, -1/12], [3/4, 1/4]], atol=1e-14)
    np.testing.assert_allclose(demo["b"], [3/4, 1/4], atol=1e-14)
    np.testing.assert_allclose(demo["D"], [[-2, 3/2, 1/2], [2, -9/2, 5/2]], atol=1e-13)
    np.testing.assert_allclose(demo["e"], [0, 0, 1], atol=1e-13)


def test_solution_satisfies_the_constraints_and_follows_the_ode(demo):
    solution = demo["solution"]
    assert solution.success, solution.message
    assert np.abs(demo["equalities"](solution.x)).max() < 1e-7
    np.testing.assert_allclose(demo["x_mesh"][0], [0, 0], atol=1e-7)
    np.testing.assert_allclose(demo["x_mesh"][-1], [np.pi, 0], atol=1e-7)
    assert np.abs(demo["x_mesh"] - demo["x_replay"]).max() < 1e-2
    # The animated replay, sampled more finely, also ends upright.
    assert abs(demo["theta"][-1] - np.pi) < 1e-2


def test_weak_motor_swings_back_before_swinging_up(demo):
    # The text describes torques at the limit and a back swing to about -54 degrees.
    u = demo["u"]
    assert np.abs(u).max() <= demo["u_max"] + 1e-9
    assert np.isclose(u[:10], -demo["u_max"], atol=1e-6).all()
    assert np.isclose(u[15:28], demo["u_max"], atol=1e-6).all()
    assert -1.0 < demo["x_replay"][:, 0].min() < -0.9


def test_both_forms_return_the_same_trajectory(demo):
    other, solution = demo["other"], demo["solution"]
    assert other.success, other.message
    np.testing.assert_allclose(other.fun, solution.fun, atol=1e-6)
    np.testing.assert_allclose(other.x, solution.x, atol=1e-4)
    # The two residual functions differ, but they vanish at the same points.
    assert np.abs(demo["equalities_from_derivatives"](solution.x)).max() < 1e-6


@pytest.mark.parametrize("nodes,A,b", [
    ("[0.0]", [[0]], [1]),
    ("[1.0]", [[1]], [1]),
    ("[0.5]", [[1/2]], [1]),
    ("[0.0, 1.0]", [[0, 0], [1/2, 1/2]], [1/2, 1/2]),
    ("[0.0, 0.5, 1.0]", [[0, 0, 0], [5/24, 1/3, -1/24], [1/6, 2/3, 1/6]],
     [1/6, 2/3, 1/6]),
])
def test_changing_the_nodes_recovers_the_chapter_schemes(nodes, A, b):
    changed = run(("tau = np.array([1/3, 1.0])", f"tau = np.array({nodes})"),
                  cells=cells_through("show(\"A\", A)"))
    np.testing.assert_allclose(changed["A"], A, atol=1e-14)
    np.testing.assert_allclose(changed["b"], b, atol=1e-14)


@pytest.mark.parametrize("nodes", ["[0.0, 1.0]", "[0.0, 0.5, 1.0]"])
def test_other_node_choices_solve_the_torque_limited_problem(nodes):
    changed = run(("tau = np.array([1/3, 1.0])", f"tau = np.array({nodes})"),
                  cells=cells_through("solution = minimize("))
    assert changed["solution"].success, changed["solution"].message
    assert np.abs(changed["equalities"](changed["solution"].x)).max() < 1e-7


def test_a_strong_motor_lifts_the_pendulum_directly():
    # The third experiment: the limit is inactive and there is no back swing.
    changed = run(("T = 8.0", "T = 4.0"), ("N = 40 ", "N = 20 "), ("u_max = 0.5", "u_max = 1.5"),
                  cells=cells_through("solution = minimize("))
    assert changed["solution"].success, changed["solution"].message
    assert 1.2 < np.abs(changed["u"]).max() < 1.5
    assert changed["x_mesh"][:, 0].min() > -1e-3


def test_a_node_at_zero_stops_the_derivative_section():
    with pytest.raises(AssertionError, match="other than 0"):
        run(("tau = np.array([1/3, 1.0])", "tau = np.array([0.0, 1.0])"))


def test_saved_outputs_show_the_hand_calculated_rule():
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    printed = "".join(
        "".join(output.get("text", ""))
        for cell in cells if cell["cell_type"] == "code"
        for output in cell["outputs"] if output["output_type"] == "stream"
    )
    assert "A = [[5/12, -1/12], [3/4, 1/4]]" in printed
    assert "b = [3/4, 1/4]" in printed
    assert "D = [[-2, 3/2, 1/2], [2, -9/2, 5/2]]" in printed


def test_notebook_keeps_colab_layout_and_folds_only_figure_cells():
    notebook = json.loads(NOTEBOOK.read_text())
    assert (notebook["nbformat"], notebook["nbformat_minor"]) == (4, 0)
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    folded = [cell for cell in notebook["cells"] if "cellView" in cell["metadata"]]
    assert len(folded) == 5
    for cell in folded:
        assert cell["metadata"]["tags"] == ["hide-input"]
        assert "".join(cell["source"]).startswith(
            ("# @title Plot", "# @title Sketch", "# @title Draw", "# @title Animate"))
        assert any(kind in output.get("data", {}) for output in cell["outputs"]
                   for kind in ("image/png", "text/html"))


def test_demo_has_its_own_toc_part_and_the_chapter_links_to_it():
    root = NOTEBOOK.parents[2]
    toc = (root / "myst.yml").read_text().split("\n  toc:\n", 1)[1]
    demos = toc.split("    - title: Demos\n", 1)[1].split("\n    - title:", 1)[0]
    assert "file: lab/notebooks/08_collocation_from_nodes.ipynb" in demos
    chapter = (root / "continuous-time-collocation.md").read_text()
    headings = {
        "#" + re.sub(r"[^a-z0-9]+", "-", line.lstrip("# ").lower()).strip("-")
        for cell in json.loads(NOTEBOOK.read_text())["cells"] if cell["cell_type"] == "markdown"
        for line in "".join(cell["source"]).splitlines() if line.startswith("## ")
    }
    links = re.findall(r"\(lab/notebooks/08_collocation_from_nodes\.ipynb(#[^)]*)?\)", chapter)
    assert len(links) >= 5
    for anchor in links:
        assert anchor == "" or anchor in headings, anchor
