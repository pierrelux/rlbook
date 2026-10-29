"""PyBaMM-backed fast-charging audit with one learned resistance scale.

The browser never advances this model.  This module runs the one-RC PyBaMM
Thevenin plant offline, applies a transparent local current governor, and
serializes immutable trajectories for the textbook replay.

Textbook currents use the charge-positive convention.  PyBaMM uses positive
current for discharge, so every plant input is the negative of the action
reported here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


RunName = Literal[
    "fresh_nominal",
    "high_resistance_stale",
    "high_resistance_calibrated",
]

RUN_ORDER: tuple[RunName, ...] = (
    "fresh_nominal",
    "high_resistance_stale",
    "high_resistance_calibrated",
)
RUN_LABELS: Mapping[RunName, str] = {
    "fresh_nominal": "Fresh plant, nominal model",
    "high_resistance_stale": "High resistance, stale model",
    "high_resistance_calibrated": "High resistance, fitted model",
}
RUN_VERDICTS: Mapping[RunName, str] = {
    "fresh_nominal": "stands",
    "high_resistance_stale": "withdrawn",
    "high_resistance_calibrated": "stands",
}
RESISTANCE_FIT_BOUNDS = (0.7, 2.5)

PAPER = "#F6F7F4"
INK = "#1B2430"
TEAL = "#2F6F8F"
MUTED = "#5C6874"
RULE = "#D2D9D7"
STANDS = "#2E7D5B"
CAVEAT = "#B8860B"
WITHDRAWN = "#A83A32"

RUN_STYLES: Mapping[RunName, tuple[str, str]] = {
    "fresh_nominal": (STANDS, "--"),
    "high_resistance_stale": (WITHDRAWN, ":"),
    "high_resistance_calibrated": (STANDS, "-"),
}

FIGURE_STYLE = {
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "font.family": "IBM Plex Sans",
    "font.sans-serif": ["IBM Plex Sans"],
    "font.monospace": ["IBM Plex Mono"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.65,
    "lines.linewidth": 1.45,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "svg.fonttype": "none",
    "svg.hashsalt": "battery-fast-charging-v1",
}

FONT_DIRECTORY = Path(__file__).resolve().parents[1] / "_static" / "battery" / "fonts"
FIGURE_FONT_FILES = (
    FONT_DIRECTORY / "IBMPlexSans-Regular.ttf",
    FONT_DIRECTORY / "IBMPlexSans-SemiBold.ttf",
    FONT_DIRECTORY / "IBMPlexMono-Regular.ttf",
    FONT_DIRECTORY / "Newsreader.ttf",
)


def _register_figure_fonts() -> None:
    """Register the vendored, licensed fonts used by the committed figure."""

    missing = [path for path in FIGURE_FONT_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing vendored battery figure fonts: "
            + ", ".join(str(path) for path in missing)
        )
    for path in FIGURE_FONT_FILES:
        font_manager.fontManager.addfont(path)


@dataclass(frozen=True)
class BatteryScenario:
    """Physical parameters, constraints, and sampling for one matched audit."""

    capacity_ah: float = 5.0
    initial_soc: float = 0.20
    target_soc: float = 0.80
    initial_temperature_c: float = 25.0
    ambient_temperature_c: float = 25.0
    current_limit_a: float = 10.0
    voltage_limit_v: float = 4.20
    temperature_limit_c: float = 35.0
    voltage_guard_v: float = 4.17
    temperature_guard_c: float = 34.5
    nominal_r0_ohm: float = 0.015
    nominal_r1_ohm: float = 0.010
    nominal_c1_f: float = 2400.0
    high_resistance_scale: float = 1.8
    cell_thermal_mass_j_per_k: float = 180.0
    cell_jig_heat_transfer_w_per_k: float = 0.55
    jig_thermal_mass_j_per_k: float = 400.0
    jig_air_heat_transfer_w_per_k: float = 0.35
    control_period_s: float = 1.0
    duration_s: float = 1800.0
    solver_rtol: float = 1e-8
    solver_atol: float = 1e-9
    fit_seed: int = 11
    fit_voltage_noise_std_v: float = 0.001
    fit_sample_period_s: float = 0.5

    def validate(self) -> None:
        if not 0.0 <= self.initial_soc < self.target_soc <= 1.0:
            raise ValueError("SOC values must satisfy 0 <= initial < target <= 1")
        positive = {
            "capacity_ah": self.capacity_ah,
            "current_limit_a": self.current_limit_a,
            "voltage_limit_v": self.voltage_limit_v,
            "temperature_limit_c": self.temperature_limit_c,
            "nominal_r0_ohm": self.nominal_r0_ohm,
            "nominal_r1_ohm": self.nominal_r1_ohm,
            "nominal_c1_f": self.nominal_c1_f,
            "high_resistance_scale": self.high_resistance_scale,
            "cell_thermal_mass_j_per_k": self.cell_thermal_mass_j_per_k,
            "cell_jig_heat_transfer_w_per_k": (
                self.cell_jig_heat_transfer_w_per_k
            ),
            "jig_thermal_mass_j_per_k": self.jig_thermal_mass_j_per_k,
            "jig_air_heat_transfer_w_per_k": self.jig_air_heat_transfer_w_per_k,
            "control_period_s": self.control_period_s,
            "duration_s": self.duration_s,
            "solver_rtol": self.solver_rtol,
            "solver_atol": self.solver_atol,
            "fit_sample_period_s": self.fit_sample_period_s,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.voltage_guard_v > self.voltage_limit_v:
            raise ValueError("voltage guard cannot exceed the plant limit")
        if self.temperature_guard_c > self.temperature_limit_c:
            raise ValueError("temperature guard cannot exceed the plant limit")
        if self.initial_temperature_c >= self.temperature_guard_c:
            raise ValueError("initial temperature must be below the controller guard")
        if self.fit_voltage_noise_std_v < 0.0:
            raise ValueError("fit_voltage_noise_std_v must be nonnegative")
        count = self.duration_s / self.control_period_s
        if not np.isclose(count, round(count), atol=1e-12, rtol=0.0):
            raise ValueError("duration_s must be a multiple of control_period_s")

    @property
    def nominal_time_constant_s(self) -> float:
        return self.nominal_r1_ohm * self.nominal_c1_f


@dataclass(frozen=True)
class BatteryMetrics:
    """Constraint, timing, and conservation diagnostics for one charge."""

    target_time_s: float
    max_voltage_v: float
    voltage_violation_duration_s: float
    max_cell_temperature_c: float
    temperature_violation_duration_s: float
    max_jig_temperature_c: float
    mean_current_a: float
    charged_capacity_ah: float
    charge_balance_error_ah: float
    minimum_voltage_margin_v: float
    minimum_temperature_margin_c: float


@dataclass(frozen=True)
class BatteryTrace:
    """One PyBaMM plant rollout under a named model/controller pairing."""

    name: RunName
    label: str
    verdict: str
    plant_resistance_scale: float
    model_resistance_scale: float
    time_s: np.ndarray
    soc: np.ndarray
    rc_overpotential_v: np.ndarray
    current_a: np.ndarray
    terminal_voltage_v: np.ndarray
    cell_temperature_c: np.ndarray
    jig_temperature_c: np.ndarray
    metrics: BatteryMetrics
    first_taper_time_s: float | None
    first_violation_time_s: float | None

    def validate(self, scenario: BatteryScenario) -> None:
        arrays = (
            self.soc,
            self.rc_overpotential_v,
            self.current_a,
            self.terminal_voltage_v,
            self.cell_temperature_c,
            self.jig_temperature_c,
        )
        if self.time_s.ndim != 1 or self.time_s.size < 2:
            raise ValueError("time_s must be a one-dimensional trajectory")
        if any(array.shape != self.time_s.shape for array in arrays):
            raise ValueError("all BatteryTrace arrays must share one time grid")
        if not all(np.all(np.isfinite(array)) for array in (self.time_s, *arrays)):
            raise ValueError("BatteryTrace contains a non-finite sample")
        if np.any(np.diff(self.time_s) <= 0.0):
            raise ValueError("BatteryTrace times must increase strictly")
        if np.min(self.soc) < -1e-9 or np.max(self.soc) > 1.0 + 1e-9:
            raise ValueError("state of charge left [0, 1]")
        if np.min(self.current_a) < -1e-9:
            raise ValueError("textbook charging current cannot be negative")
        if np.max(self.current_a) > scenario.current_limit_a + 1e-8:
            raise ValueError("charging current exceeded its action bound")


@dataclass(frozen=True)
class DiagnosticTrace:
    """Separate pulse record used to fit the resistance multiplier."""

    time_s: np.ndarray
    current_a: np.ndarray
    soc: np.ndarray
    clean_voltage_v: np.ndarray
    measured_voltage_v: np.ndarray
    resistance_scale: float
    seed: int
    noise_std_v: float


@dataclass(frozen=True)
class ResistanceFit:
    """Least-squares estimate and voltage residual for the pulse record."""

    resistance_scale: float
    voltage_rmse_v: float
    informative_samples: int


@dataclass(frozen=True)
class _BatteryState:
    soc: float
    rc_overpotential_v: float
    cell_temperature_c: float
    jig_temperature_c: float


def open_circuit_voltage(soc: Any) -> Any:
    """Teaching OCV curve: ``3.10 + z + 0.08 tanh((z-0.50)/0.10)`` V."""

    if isinstance(soc, (int, float, np.ndarray, np.generic)):
        transition = np.tanh((soc - 0.50) / 0.10)
    else:
        # PyBaMM symbols deliberately stay behind the optional dependency.
        import pybamm

        transition = pybamm.tanh((soc - 0.50) / 0.10)
    return 3.10 + soc + 0.08 * transition


def pybamm_current(charge_current_a: float | np.ndarray) -> float | np.ndarray:
    """Convert the textbook charge-positive current to PyBaMM's convention."""

    return -np.asarray(charge_current_a) if isinstance(charge_current_a, np.ndarray) else -float(charge_current_a)


def resistance_parameters(
    scenario: BatteryScenario, resistance_scale: float
) -> dict[str, float]:
    """Return the scaled one-RC values while preserving the RC time constant."""

    if not np.isfinite(resistance_scale) or resistance_scale <= 0.0:
        raise ValueError("resistance_scale must be finite and positive")
    return {
        "r0_ohm": scenario.nominal_r0_ohm * resistance_scale,
        "r1_ohm": scenario.nominal_r1_ohm * resistance_scale,
        "c1_f": scenario.nominal_c1_f / resistance_scale,
    }


def _make_pybamm_model_and_parameters(
    scenario: BatteryScenario,
    resistance_scale: float,
    current: Any = 0.0,
) -> tuple[Any, Any, Any]:
    """Construct a PyBaMM 26.6.2.0 one-RC Thevenin plant."""

    try:
        import pybamm
    except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "battery artifact generation requires `uv sync --group artifacts`"
        ) from error

    if pybamm.__version__ != "26.6.2.0":
        raise RuntimeError(
            "battery artifacts require PyBaMM 26.6.2.0; "
            f"found {pybamm.__version__}"
        )
    scenario.validate()
    scaled = resistance_parameters(scenario, resistance_scale)
    model = pybamm.equivalent_circuit.Thevenin(
        options={"number of rc elements": 1}
    )
    parameters = model.default_parameter_values.copy()
    parameters.update(
        {
            "Initial SoC": scenario.initial_soc,
            "Initial temperature [K]": scenario.initial_temperature_c + 273.15,
            "Cell capacity [A.h]": scenario.capacity_ah,
            "Nominal cell capacity [A.h]": scenario.capacity_ah,
            "Ambient temperature [K]": scenario.ambient_temperature_c + 273.15,
            "Current function [A]": current,
            # The experiment measures violations rather than terminating at them.
            "Upper voltage cut-off [V]": 5.0,
            "Lower voltage cut-off [V]": 2.5,
            "Cell thermal mass [J/K]": scenario.cell_thermal_mass_j_per_k,
            "Cell-jig heat transfer coefficient [W/K]": (
                scenario.cell_jig_heat_transfer_w_per_k
            ),
            "Jig thermal mass [J/K]": scenario.jig_thermal_mass_j_per_k,
            "Jig-air heat transfer coefficient [W/K]": (
                scenario.jig_air_heat_transfer_w_per_k
            ),
            "Open-circuit voltage [V]": open_circuit_voltage,
            "R0 [Ohm]": scaled["r0_ohm"],
            "Element-1 initial overpotential [V]": 0.0,
            "R1 [Ohm]": scaled["r1_ohm"],
            "C1 [F]": scaled["c1_f"],
            "Entropic change [V/K]": 0.0,
        }
    )
    return pybamm, model, parameters


def _make_pybamm_simulation(
    scenario: BatteryScenario, resistance_scale: float
) -> Any:
    """Construct an input-driven plant for fixed-current diagnostic stepping."""

    pybamm, model, parameters = _make_pybamm_model_and_parameters(
        scenario,
        resistance_scale,
        current=pybamm_placeholder_current(),
    )
    solver = pybamm.IDAKLUSolver(
        rtol=scenario.solver_rtol, atol=scenario.solver_atol
    )
    return pybamm.Simulation(model, parameter_values=parameters, solver=solver)


def pybamm_placeholder_current() -> Any:
    """Return the optional PyBaMM input used only by explicit stepping."""

    try:
        import pybamm
    except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "battery artifact generation requires `uv sync --group artifacts`"
        ) from error
    return pybamm.InputParameter("Current input [A]")


def _terminal_voltage(
    state: _BatteryState,
    charge_current_a: float,
    scenario: BatteryScenario,
    resistance_scale: float,
) -> float:
    scaled = resistance_parameters(scenario, resistance_scale)
    return float(
        open_circuit_voltage(state.soc)
        + charge_current_a * scaled["r0_ohm"]
        + state.rc_overpotential_v
    )


def predictive_current_governor(
    state: _BatteryState,
    scenario: BatteryScenario,
    model_resistance_scale: float,
) -> float:
    """Return the largest current inside local voltage and thermal envelopes."""

    return float(
        _governed_current(
            state.soc,
            state.rc_overpotential_v,
            state.cell_temperature_c,
            scenario,
            model_resistance_scale,
            maximum=max,
            minimum=min,
            square_root=np.sqrt,
        )
    )


def _governed_current(
    soc: Any,
    rc_overpotential_v: Any,
    cell_temperature_c: Any,
    scenario: BatteryScenario,
    model_resistance_scale: float,
    *,
    maximum: Any,
    minimum: Any,
    square_root: Any,
) -> Any:
    """Evaluate the same local envelope with numeric or PyBaMM operators."""

    scaled = resistance_parameters(scenario, model_resistance_scale)
    voltage_ceiling = maximum(
        (
            scenario.voltage_guard_v
            - open_circuit_voltage(soc)
            - rc_overpotential_v
        )
        / scaled["r0_ohm"],
        0.0,
    )
    resistance_sum = scaled["r0_ohm"] + scaled["r1_ohm"]
    thermal_headroom = maximum(
        scenario.cell_jig_heat_transfer_w_per_k
        * (scenario.temperature_guard_c - cell_temperature_c),
        0.0,
    )
    thermal_ceiling = square_root(thermal_headroom / resistance_sum)
    return minimum(
        scenario.current_limit_a,
        minimum(voltage_ceiling, thermal_ceiling),
    )


def _extract_state(solution: Any) -> _BatteryState:
    return _BatteryState(
        soc=float(solution["SoC"].entries[-1]),
        rc_overpotential_v=float(
            solution["Element-1 overpotential [V]"].entries[-1]
        ),
        cell_temperature_c=float(solution["Cell temperature [degC]"].entries[-1]),
        jig_temperature_c=float(solution["Jig temperature [degC]"].entries[-1]),
    )


def _metrics(
    time_s: np.ndarray,
    soc: np.ndarray,
    current_a: np.ndarray,
    voltage_v: np.ndarray,
    cell_temperature_c: np.ndarray,
    jig_temperature_c: np.ndarray,
    scenario: BatteryScenario,
) -> BatteryMetrics:
    dt = np.diff(time_s)
    voltage_violation_duration = threshold_duration(
        time_s, voltage_v, scenario.voltage_limit_v
    )
    temperature_violation_duration = threshold_duration(
        time_s, cell_temperature_c, scenario.temperature_limit_c
    )
    charged_ah = float(np.trapezoid(current_a, time_s) / 3600.0)
    expected_ah = float(scenario.capacity_ah * (soc[-1] - soc[0]))
    return BatteryMetrics(
        target_time_s=float(time_s[-1]),
        max_voltage_v=float(np.max(voltage_v)),
        voltage_violation_duration_s=voltage_violation_duration,
        max_cell_temperature_c=float(np.max(cell_temperature_c)),
        temperature_violation_duration_s=temperature_violation_duration,
        max_jig_temperature_c=float(np.max(jig_temperature_c)),
        mean_current_a=float(charged_ah * 3600.0 / time_s[-1]),
        charged_capacity_ah=charged_ah,
        charge_balance_error_ah=float(charged_ah - expected_ah),
        minimum_voltage_margin_v=float(scenario.voltage_limit_v - np.max(voltage_v)),
        minimum_temperature_margin_c=float(
            scenario.temperature_limit_c - np.max(cell_temperature_c)
        ),
    )


def threshold_duration(
    time_s: np.ndarray, values: np.ndarray, threshold: float
) -> float:
    """Integrate time above a threshold with linear crossing interpolation."""

    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    if time_s.ndim != 1 or values.shape != time_s.shape or time_s.size < 2:
        raise ValueError("threshold_duration expects matched one-dimensional arrays")
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("threshold_duration times must increase strictly")
    offset = values - float(threshold)
    duration = 0.0
    for left, right, step in zip(offset[:-1], offset[1:], np.diff(time_s)):
        if left > 0.0 and right > 0.0:
            duration += step
        elif left <= 0.0 < right:
            duration += step * right / (right - left)
        elif left > 0.0 >= right:
            duration += step * left / (left - right)
    return float(duration)


def simulate_charge(
    name: RunName,
    plant_resistance_scale: float,
    model_resistance_scale: float,
    scenario: BatteryScenario | None = None,
) -> BatteryTrace:
    """Run one closed-loop charge on the PyBaMM Thevenin plant."""

    scenario = scenario or BatteryScenario()
    scenario.validate()
    if name not in RUN_ORDER:
        raise ValueError(f"unknown battery audit run: {name}")
    pybamm, model, parameters = _make_pybamm_model_and_parameters(
        scenario, plant_resistance_scale
    )
    def controller(variables: Mapping[str, Any]) -> Any:
        charge_current = _governed_current(
            variables["SoC"],
            variables["Element-1 overpotential [V]"],
            variables["Cell temperature [degC]"],
            scenario,
            model_resistance_scale,
            maximum=pybamm.maximum,
            minimum=pybamm.minimum,
            square_root=pybamm.sqrt,
        )
        return -charge_current

    def target_remaining(variables: Mapping[str, Any]) -> Any:
        return scenario.target_soc - variables["SoC"]

    step = pybamm.step.CustomStepExplicit(
        controller,
        duration=scenario.duration_s,
        termination=pybamm.step.CustomTermination(
            "target state of charge", target_remaining
        ),
        period=scenario.control_period_s,
        direction="charge",
    )
    simulation = pybamm.Simulation(
        model,
        experiment=pybamm.Experiment([step]),
        parameter_values=parameters,
        solver=pybamm.IDAKLUSolver(
            rtol=scenario.solver_rtol, atol=scenario.solver_atol
        ),
    )
    solution = simulation.solve()
    time_array = np.asarray(solution["Time [s]"].entries, dtype=float)
    soc_array = np.asarray(solution["SoC"].entries, dtype=float)
    current_array = -np.asarray(solution["Current [A]"].entries, dtype=float)
    voltage_array = np.asarray(solution["Voltage [V]"].entries, dtype=float)
    cell_array = np.asarray(
        solution["Cell temperature [degC]"].entries, dtype=float
    )
    jig_array = np.asarray(
        solution["Jig temperature [degC]"].entries, dtype=float
    )
    rc_array = np.asarray(
        solution["Element-1 overpotential [V]"].entries, dtype=float
    )
    if soc_array[-1] < scenario.target_soc - 2e-7:
        raise RuntimeError("battery charge did not reach the target within duration_s")
    metrics = _metrics(
        time_array,
        soc_array,
        current_array,
        voltage_array,
        cell_array,
        jig_array,
        scenario,
    )
    interval_current = current_array[:-1]
    taper_indices = np.flatnonzero(
        interval_current < scenario.current_limit_a - 1e-6
    )
    violating_indices = np.flatnonzero(
        (voltage_array > scenario.voltage_limit_v + 1e-10)
        | (cell_array > scenario.temperature_limit_c + 1e-10)
    )
    first_violation = None
    if violating_indices.size:
        first_index = int(violating_indices[0])
        if first_index == 0:
            first_violation = float(time_array[0])
        else:
            left_time = time_array[first_index - 1]
            right_time = time_array[first_index]
            left_offset = voltage_array[first_index - 1] - scenario.voltage_limit_v
            right_offset = voltage_array[first_index] - scenario.voltage_limit_v
            if right_offset > 0.0 and left_offset <= 0.0:
                fraction = -left_offset / (right_offset - left_offset)
                first_violation = float(
                    left_time + fraction * (right_time - left_time)
                )
            else:
                first_violation = float(right_time)
    trace = BatteryTrace(
        name=name,
        label=RUN_LABELS[name],
        verdict=RUN_VERDICTS[name],
        plant_resistance_scale=float(plant_resistance_scale),
        model_resistance_scale=float(model_resistance_scale),
        time_s=time_array,
        soc=soc_array,
        rc_overpotential_v=rc_array,
        current_a=current_array,
        terminal_voltage_v=voltage_array,
        cell_temperature_c=cell_array,
        jig_temperature_c=jig_array,
        metrics=metrics,
        first_taper_time_s=(
            float(time_array[taper_indices[0]]) if taper_indices.size else None
        ),
        first_violation_time_s=first_violation,
    )
    trace.validate(scenario)
    return trace


def _pulse_current(time_s: float) -> float:
    return 5.0 if 20.0 <= time_s < 30.0 else 0.0


def simulate_diagnostic_pulse(
    scenario: BatteryScenario | None = None,
    resistance_scale: float | None = None,
) -> DiagnosticTrace:
    """Generate the separate 20 s rest, 10 s pulse, 40 s rest record."""

    scenario = scenario or BatteryScenario()
    scenario.validate()
    scale = (
        scenario.high_resistance_scale
        if resistance_scale is None
        else resistance_scale
    )
    simulation = _make_pybamm_simulation(scenario, scale)
    state = _BatteryState(
        scenario.initial_soc,
        0.0,
        scenario.initial_temperature_c,
        scenario.initial_temperature_c,
    )
    period = scenario.fit_sample_period_s
    time_grid = np.arange(0.0, 70.0 + 0.5 * period, period)
    current = np.asarray([_pulse_current(value) for value in time_grid])
    soc = np.empty_like(time_grid)
    voltage = np.empty_like(time_grid)
    for index, time_s in enumerate(time_grid):
        soc[index] = state.soc
        voltage[index] = _terminal_voltage(state, current[index], scenario, scale)
        if index + 1 < time_grid.size:
            solution = simulation.step(
                period,
                inputs={"Current input [A]": pybamm_current(current[index])},
                save=False,
            )
            state = _extract_state(solution)
    noise = np.random.default_rng(scenario.fit_seed).normal(
        0.0, scenario.fit_voltage_noise_std_v, size=time_grid.size
    )
    return DiagnosticTrace(
        time_s=time_grid,
        current_a=current,
        soc=soc,
        clean_voltage_v=voltage,
        measured_voltage_v=voltage + noise,
        resistance_scale=float(scale),
        seed=scenario.fit_seed,
        noise_std_v=scenario.fit_voltage_noise_std_v,
    )


def fit_resistance_scale(
    trace: DiagnosticTrace,
    scenario: BatteryScenario | None = None,
) -> ResistanceFit:
    """Fit the resistance multiplier by linear least squares on pulse voltage."""

    scenario = scenario or BatteryScenario()
    scenario.validate()
    if not (
        trace.time_s.shape
        == trace.current_a.shape
        == trace.soc.shape
        == trace.measured_voltage_v.shape
    ):
        raise ValueError("diagnostic arrays must share one time grid")
    dt = np.diff(trace.time_s)
    if np.any(dt <= 0.0):
        raise ValueError("diagnostic times must increase strictly")
    base_rc = 0.0
    feature = np.empty_like(trace.time_s)
    tau = scenario.nominal_time_constant_s
    for index, charge_current in enumerate(trace.current_a):
        feature[index] = charge_current * scenario.nominal_r0_ohm + base_rc
        if index + 1 < trace.time_s.size:
            decay = float(np.exp(-dt[index] / tau))
            base_rc = (
                decay * base_rc
                + (1.0 - decay) * charge_current * scenario.nominal_r1_ohm
            )
    centered_voltage = trace.measured_voltage_v - np.asarray(
        open_circuit_voltage(trace.soc), dtype=float
    )
    informative = np.abs(feature) > 1e-8
    if np.count_nonzero(informative) < 2:
        raise ValueError("diagnostic pulse contains no resistance information")
    x = feature[informative]
    y = centered_voltage[informative]
    unconstrained_scale = float(np.dot(x, y) / np.dot(x, x))
    scale = float(np.clip(unconstrained_scale, *RESISTANCE_FIT_BOUNDS))
    fitted_voltage = np.asarray(open_circuit_voltage(trace.soc), dtype=float) + scale * feature
    rmse = float(np.sqrt(np.mean((trace.measured_voltage_v - fitted_voltage) ** 2)))
    return ResistanceFit(scale, rmse, int(np.count_nonzero(informative)))


def run_battery_audit(
    scenario: BatteryScenario | None = None,
) -> tuple[dict[RunName, BatteryTrace], DiagnosticTrace, ResistanceFit]:
    """Fit one parameter, then run the three matched PyBaMM comparisons."""

    scenario = scenario or BatteryScenario()
    diagnostic = simulate_diagnostic_pulse(scenario)
    fitted = fit_resistance_scale(diagnostic, scenario)
    results: dict[RunName, BatteryTrace] = {
        "fresh_nominal": simulate_charge("fresh_nominal", 1.0, 1.0, scenario),
        "high_resistance_stale": simulate_charge(
            "high_resistance_stale",
            scenario.high_resistance_scale,
            1.0,
            scenario,
        ),
        "high_resistance_calibrated": simulate_charge(
            "high_resistance_calibrated",
            scenario.high_resistance_scale,
            fitted.resistance_scale,
            scenario,
        ),
    }
    return results, diagnostic, fitted


def _replay_indices(
    trace: BatteryTrace,
    frame_stride: int,
    scenario: BatteryScenario,
) -> np.ndarray:
    indices = set(range(0, trace.time_s.size, frame_stride))
    indices.add(trace.time_s.size - 1)
    for event_time in (trace.first_taper_time_s, trace.first_violation_time_s):
        if event_time is not None:
            event_index = int(np.argmin(np.abs(trace.time_s - event_time)))
            indices.update(
                index
                for index in (event_index - 1, event_index, event_index + 1)
                if 0 <= index < trace.time_s.size
            )
    violating = np.flatnonzero(
        trace.terminal_voltage_v > scenario.voltage_limit_v + 1e-10
    )
    if violating.size:
        indices.add(int(violating[np.argmax(trace.terminal_voltage_v[violating])]))
    return np.asarray(sorted(indices), dtype=int)


def audit_to_artifact(
    results: Mapping[RunName, BatteryTrace],
    diagnostic: DiagnosticTrace,
    fitted: ResistanceFit,
    scenario: BatteryScenario,
    frame_stride: int = 5,
) -> dict[str, object]:
    """Serialize the strict schema-versioned browser replay contract."""

    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    runs: dict[str, object] = {}
    for name in RUN_ORDER:
        trace = results[name]
        indices = _replay_indices(trace, frame_stride, scenario)
        runs[name] = {
            "label": trace.label,
            "verdict": trace.verdict,
            "style": {
                "color": RUN_STYLES[name][0],
                "dash": RUN_STYLES[name][1],
            },
            "plant_resistance_scale": trace.plant_resistance_scale,
            "model_resistance_scale": trace.model_resistance_scale,
            "frames": [
                {
                    "time_s": float(trace.time_s[index]),
                    "soc": float(trace.soc[index]),
                    "current_a": float(trace.current_a[index]),
                    "terminal_voltage_v": float(
                        trace.terminal_voltage_v[index]
                    ),
                    "cell_temperature_c": float(
                        trace.cell_temperature_c[index]
                    ),
                    "jig_temperature_c": float(
                        trace.jig_temperature_c[index]
                    ),
                    "rc_overpotential_v": float(
                        trace.rc_overpotential_v[index]
                    ),
                }
                for index in indices
            ],
            "metrics": asdict(trace.metrics),
            "events": {
                "first_taper_time_s": trace.first_taper_time_s,
                "first_violation_time_s": trace.first_violation_time_s,
                "target_time_s": trace.metrics.target_time_s,
            },
        }
    return {
        "schema_version": 1,
        "title": "Fast charging when resistance drifts",
        "description": (
            "Recorded PyBaMM trajectories under one current governor. The plant "
            "and controller differ only through the stated resistance scale."
        ),
        "playback_fps": 25,
        "frame_stride": frame_stride,
        "scenario": asdict(scenario),
        "model": {
            "library": "PyBaMM",
            "version": "26.6.2.0",
            "name": "Thevenin Equivalent Circuit Model",
            "options": {
                "number of rc elements": 1,
                "diffusion element": "false",
                "operating mode": "current",
            },
            "ocv_expression_v": "3.10 + soc + 0.08 * tanh((soc - 0.50) / 0.10)",
            "pybamm_current_convention": "positive discharge",
            "textbook_current_convention": "positive charge",
        },
        "diagnostic": {
            "protocol": "20 s rest, 10 s at 5 A charge, 40 s rest",
            "seed": diagnostic.seed,
            "noise_std_v": diagnostic.noise_std_v,
            "true_resistance_scale": diagnostic.resistance_scale,
            "fitted_resistance_scale": fitted.resistance_scale,
            "voltage_rmse_v": fitted.voltage_rmse_v,
            "informative_samples": fitted.informative_samples,
            "fit_bounds": list(RESISTANCE_FIT_BOUNDS),
        },
        "runs": runs,
        "limitations": [
            "The high-resistance plant is a controlled counterfactual, not a general aging law.",
            "The OCV curve and thermal constants define a teaching cell, not a commercial product.",
            "The limits are experimental constraints, not manufacturer-approved charging limits.",
            "The local current governor is not an electrochemical safety certification.",
            (
                "The 35 degree C plant bound is never reached, while the "
                "conservative 34.5 degree C local thermal-headroom envelope "
                "does limit requested current."
            ),
        ],
    }


def save_artifact(artifact: Mapping[str, object], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["left"].set_color(RULE)
    axis.spines["bottom"].set_color(RULE)
    axis.tick_params(colors=MUTED)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)
    axis.title.set_color(INK)
    axis.grid(axis="y", color=RULE, linewidth=0.45, alpha=0.55)


def _draw_status_card(
    axis: plt.Axes,
    trace: BatteryTrace,
    scenario: BatteryScenario,
) -> None:
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    color, linestyle = RUN_STYLES[trace.name]
    axis.add_patch(
        FancyBboxPatch(
            (0.02, 0.06),
            0.96,
            0.88,
            boxstyle="round,pad=0.015,rounding_size=0.035",
            facecolor=PAPER,
            edgecolor=RULE,
            linewidth=0.8,
        )
    )
    axis.add_patch(
        FancyBboxPatch(
            (0.09, 0.28),
            0.25,
            0.46,
            boxstyle="round,pad=0.01,rounding_size=0.025",
            facecolor="none",
            edgecolor=INK,
            linewidth=1.0,
        )
    )
    axis.add_patch(
        Rectangle((0.34, 0.42), 0.035, 0.18, facecolor=INK, edgecolor=INK)
    )
    fill_height = 0.40 * float(trace.soc[-1])
    axis.add_patch(
        Rectangle(
            (0.115, 0.305),
            0.20,
            fill_height,
            facecolor=color,
            edgecolor="none",
            alpha=0.88,
        )
    )
    wrapped_label = trace.label.replace(", ", ",\n", 1)
    axis.text(
        0.44,
        0.80,
        wrapped_label,
        color=INK,
        fontsize=7.4,
        va="top",
        linespacing=1.08,
    )
    verdict = "within bounds" if trace.verdict == "stands" else "voltage bound crossed"
    axis.text(0.44, 0.50, verdict, color=color, fontsize=7.5, va="center")
    axis.plot([0.44, 0.58], [0.30, 0.30], color=color, linestyle=linestyle)
    axis.text(
        0.62,
        0.30,
        f"{trace.metrics.target_time_s / 60:.1f} min",
        color=INK,
        fontsize=7.3,
        va="center",
        fontfamily="monospace",
    )


def make_summary_figure(
    results: Mapping[RunName, BatteryTrace],
    scenario: BatteryScenario,
) -> plt.Figure:
    """Create the matched vector fallback and evidence table."""

    missing = [name for name in RUN_ORDER if name not in results]
    if missing:
        raise ValueError(f"missing battery runs: {missing}")
    _register_figure_fonts()
    with mpl.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.2, 5.55), constrained_layout=True)
        grid = figure.add_gridspec(
            4,
            3,
            height_ratios=(0.70, 1.0, 1.0, 0.73),
            hspace=0.18,
        )
        for column, name in enumerate(RUN_ORDER):
            _draw_status_card(figure.add_subplot(grid[0, column]), results[name], scenario)

        axes = {
            "voltage": figure.add_subplot(grid[1, :2]),
            "soc": figure.add_subplot(grid[1, 2]),
            "current": figure.add_subplot(grid[2, :2]),
            "temperature": figure.add_subplot(grid[2, 2]),
        }
        for name in RUN_ORDER:
            trace = results[name]
            color, linestyle = RUN_STYLES[name]
            minutes = trace.time_s / 60.0
            axes["voltage"].plot(
                minutes, trace.terminal_voltage_v, color=color, linestyle=linestyle
            )
            axes["soc"].plot(minutes, 100.0 * trace.soc, color=color, linestyle=linestyle)
            axes["current"].plot(minutes, trace.current_a, color=color, linestyle=linestyle)
            axes["temperature"].plot(
                minutes, trace.cell_temperature_c, color=color, linestyle=linestyle
            )

        axes["voltage"].axhline(
            scenario.voltage_limit_v, color=WITHDRAWN, linewidth=0.8, alpha=0.8
        )
        axes["voltage"].axhline(
            scenario.voltage_guard_v, color=TEAL, linestyle="--", linewidth=0.75
        )
        stale = results["high_resistance_stale"]
        peak = int(np.argmax(stale.terminal_voltage_v))
        axes["voltage"].annotate(
            f"stale peak {stale.metrics.max_voltage_v:.3f} V",
            xy=(stale.time_s[peak] / 60.0, stale.terminal_voltage_v[peak]),
            xytext=(stale.time_s[peak] / 60.0 + 1.1, 4.31),
            color=WITHDRAWN,
            fontsize=7.3,
            arrowprops={"arrowstyle": "-", "color": WITHDRAWN, "linewidth": 0.7},
        )
        axes["voltage"].text(
            24.8,
            scenario.voltage_limit_v + 0.025,
            "plant bound",
            color=WITHDRAWN,
            fontsize=7.0,
            ha="right",
        )
        axes["voltage"].text(
            0.3,
            scenario.voltage_guard_v - 0.035,
            "model guard",
            color=TEAL,
            fontsize=7.0,
        )
        axes["soc"].axhline(80.0, color=TEAL, linestyle="--", linewidth=0.75)
        axes["soc"].text(0.35, 81.2, "target", color=TEAL, fontsize=7.0)
        axes["current"].axhline(
            scenario.current_limit_a, color=TEAL, linestyle="--", linewidth=0.75
        )
        axes["temperature"].axhline(
            scenario.temperature_limit_c, color=WITHDRAWN, linewidth=0.8
        )
        axes["temperature"].text(
            0.3, scenario.temperature_limit_c - 1.0, "35 °C bound", color=WITHDRAWN, fontsize=7.0
        )

        axes["voltage"].set(title="Terminal voltage", ylabel="voltage (V)")
        axes["soc"].set(title="Stored charge", ylabel="SOC (%)")
        axes["current"].set(
            title="Governor action", xlabel="time (min)", ylabel="charge current (A)"
        )
        axes["temperature"].set(
            title="Cell temperature", xlabel="time (min)", ylabel="temperature (°C)"
        )
        longest_minutes = max(trace.time_s[-1] for trace in results.values()) / 60.0
        for axis in axes.values():
            axis.set_xlim(0.0, longest_minutes)
            _style_axis(axis)
        axes["voltage"].set_ylim(3.15, 4.30)
        axes["soc"].set_ylim(18.0, 83.0)
        axes["current"].set_ylim(-0.25, 10.7)
        axes["temperature"].set_ylim(24.7, scenario.temperature_limit_c + 0.5)

        table_axis = figure.add_subplot(grid[3, :])
        table_axis.axis("off")
        cell_text = [
            [
                results[name].label,
                f"{results[name].metrics.target_time_s / 60:.1f}",
                f"{results[name].metrics.max_voltage_v:.3f}",
                f"{results[name].metrics.voltage_violation_duration_s:.0f}",
                f"{results[name].metrics.max_cell_temperature_c:.1f}",
            ]
            for name in RUN_ORDER
        ]
        table = table_axis.table(
            cellText=cell_text,
            colLabels=(
                "plant and controller model",
                "target (min)",
                "peak V",
                "time >4.20 V (s)",
                "peak cell °C",
            ),
            colWidths=(0.41, 0.14, 0.12, 0.19, 0.14),
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.4)
        table.scale(1.0, 1.25)
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor(RULE)
            cell.set_linewidth(0.55)
            cell.set_facecolor(PAPER)
            cell.get_text().set_color(INK)
            if row == 0:
                cell.get_text().set_fontweight("semibold")
            if row > 0 and column > 0:
                cell.get_text().set_fontfamily("monospace")
            if row == 2:
                cell.get_text().set_color(WITHDRAWN)

        figure.suptitle(
            "One fitted parameter restores the tested voltage margin",
            color=INK,
            fontsize=14,
            fontfamily="Newsreader",
            fontweight="normal",
        )
        return figure


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "BatteryMetrics",
    "BatteryScenario",
    "BatteryTrace",
    "DiagnosticTrace",
    "ResistanceFit",
    "RUN_LABELS",
    "RUN_ORDER",
    "RUN_STYLES",
    "RESISTANCE_FIT_BOUNDS",
    "RunName",
    "audit_to_artifact",
    "fit_resistance_scale",
    "make_summary_figure",
    "open_circuit_voltage",
    "predictive_current_governor",
    "pybamm_current",
    "resistance_parameters",
    "run_battery_audit",
    "save_artifact",
    "sha256",
    "simulate_charge",
    "simulate_diagnostic_pulse",
    "threshold_duration",
]
