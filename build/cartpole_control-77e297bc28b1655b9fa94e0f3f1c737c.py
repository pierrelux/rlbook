"""Cart-pole trajectory optimization and local LQR demonstrations.

The input is horizontal acceleration of the cart.  This choice makes the
action channel visible in the equations and matches the classroom analogy of
moving a finger under a pen.  Direct transcription and single shooting share
one scenario exactly.  The LQR experiment reuses the same continuous plant and
physical limits at the finer sampling interval needed for local feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import Bounds, minimize


OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
}

DIRECT_COLOR = OI["blue"]
SHOOTING_COLOR = OI["vermilion"]
UNCONTROLLED_COLOR = "0.42"

FIGURE_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.65,
    "lines.linewidth": 1.5,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
}


@dataclass(frozen=True)
class CartPoleParameters:
    """Physical parameters and limits for the teaching plant."""

    gravity: float = 9.81
    pole_length: float = 0.55
    angular_damping: float = 0.08
    rail_limit: float = 2.40
    velocity_limit: float = 4.00
    acceleration_limit: float = 8.00
    angular_velocity_limit: float = 12.0


@dataclass(frozen=True)
class SwingUpScenario:
    """One matched nonlinear program used by both numerical formulations."""

    parameters: CartPoleParameters = CartPoleParameters()
    step_size: float = 0.15
    horizon_steps: int = 30
    initial_state: tuple[float, float, float, float] = (0.0, 0.0, np.pi, 0.0)
    state_weights: tuple[float, float, float, float] = (0.05, 0.01, 0.25, 0.01)
    control_weight: float = 0.004
    terminal_weights: tuple[float, float, float, float] = (20.0, 5.0, 120.0, 12.0)

    @property
    def duration(self) -> float:
        return self.step_size * self.horizon_steps


@dataclass(frozen=True)
class TrajectoryResult:
    """Solution and diagnostics for one trajectory-optimization formulation."""

    method: str
    time: np.ndarray
    state: np.ndarray
    control: np.ndarray
    objective: float
    success: bool
    iterations: int
    dynamics_defect: float
    message: str
    decision_variables: int
    dynamics_equalities: int


@dataclass(frozen=True)
class LQRDesign:
    """Discrete linearization and infinite-horizon LQR solution."""

    state_matrix: np.ndarray
    input_matrix: np.ndarray
    cost_matrix: np.ndarray
    control_cost: np.ndarray
    riccati_matrix: np.ndarray
    gain: np.ndarray
    closed_loop_eigenvalues: np.ndarray


@dataclass(frozen=True)
class ClosedLoopResult:
    """One nonlinear validation rollout for an LQR or zero controller."""

    name: str
    initial_angle_deg: float
    time: np.ndarray
    state: np.ndarray
    control: np.ndarray
    terminated: bool
    termination_reason: str | None

    @property
    def final_angle_error_deg(self) -> float:
        return float(np.rad2deg(wrap_angle(self.state[-1, 2])))

    @property
    def stabilized(self) -> bool:
        return bool(
            not self.terminated
            and abs(self.final_angle_error_deg) < 1.0
            and abs(self.state[-1, 3]) < 0.10
        )


@dataclass(frozen=True)
class OpenLoopReplay:
    """Nominal and disturbed realizations of one fixed control sequence."""

    time: np.ndarray
    nominal_state: np.ndarray
    disturbed_state: np.ndarray
    command: np.ndarray
    disturbance: np.ndarray


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """Map angles to ``[-pi, pi]`` without changing the simulated state."""

    return np.arctan2(np.sin(angle), np.cos(angle))


def cartpole_dynamics(
    state: np.ndarray,
    acceleration: float,
    parameters: CartPoleParameters = CartPoleParameters(),
) -> np.ndarray:
    """Continuous nonlinear dynamics with angle measured from upright."""

    position, velocity, angle, angular_velocity = np.asarray(state, dtype=float)
    del position
    angular_acceleration = (
        parameters.gravity * np.sin(angle)
        - acceleration * np.cos(angle)
    ) / parameters.pole_length - parameters.angular_damping * angular_velocity
    return np.array(
        [velocity, acceleration, angular_velocity, angular_acceleration],
        dtype=float,
    )


def rk4_step(
    state: np.ndarray,
    acceleration: float,
    parameters: CartPoleParameters = CartPoleParameters(),
    step_size: float = 0.02,
) -> np.ndarray:
    """Advance the nonlinear plant by one zero-order-hold RK4 step."""

    state = np.asarray(state, dtype=float)
    k1 = cartpole_dynamics(state, acceleration, parameters)
    k2 = cartpole_dynamics(state + 0.5 * step_size * k1, acceleration, parameters)
    k3 = cartpole_dynamics(state + 0.5 * step_size * k2, acceleration, parameters)
    k4 = cartpole_dynamics(state + step_size * k3, acceleration, parameters)
    return state + (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout_controls(
    controls: np.ndarray,
    scenario: SwingUpScenario = SwingUpScenario(),
) -> np.ndarray:
    """Simulate the shared swing-up model from its fixed initial state."""

    controls = np.asarray(controls, dtype=float)
    if controls.shape != (scenario.horizon_steps,):
        raise ValueError("controls must have one scalar per horizon step")
    state = np.empty((scenario.horizon_steps + 1, 4), dtype=float)
    state[0] = np.asarray(scenario.initial_state, dtype=float)
    for index, acceleration in enumerate(controls):
        state[index + 1] = rk4_step(
            state[index],
            float(acceleration),
            scenario.parameters,
            scenario.step_size,
        )
    return state


def swingup_objective(
    state: np.ndarray,
    control: np.ndarray,
    scenario: SwingUpScenario = SwingUpScenario(),
) -> float:
    """Return the common running and terminal cost for swing-up."""

    state = np.asarray(state, dtype=float)
    control = np.asarray(control, dtype=float)
    position, velocity, angle, angular_velocity = state[:-1].T
    q_position, q_velocity, q_angle, q_angular_velocity = scenario.state_weights
    running = scenario.step_size * np.sum(
        q_position * position**2
        + q_velocity * velocity**2
        + q_angle * (1.0 - np.cos(angle))
        + q_angular_velocity * angular_velocity**2
        + scenario.control_weight * control**2
    )
    terminal = state[-1]
    f_position, f_velocity, f_angle, f_angular_velocity = scenario.terminal_weights
    terminal_cost = (
        f_position * terminal[0] ** 2
        + f_velocity * terminal[1] ** 2
        + f_angle * (1.0 - np.cos(terminal[2]))
        + f_angular_velocity * terminal[3] ** 2
    )
    return float(running + terminal_cost)


def initial_control_guess(scenario: SwingUpScenario = SwingUpScenario()) -> np.ndarray:
    """Give both solvers the same damped resonant acceleration sequence."""

    time = scenario.step_size * np.arange(scenario.horizon_steps)
    frequency = np.sqrt(scenario.parameters.gravity / scenario.parameters.pole_length)
    guess = 5.0 * np.sin(frequency * time + np.pi / 2.0) * np.exp(-0.05 * time)
    return np.clip(
        guess,
        -scenario.parameters.acceleration_limit,
        scenario.parameters.acceleration_limit,
    )


def _state_constraint_margin(state: np.ndarray, parameters: CartPoleParameters) -> np.ndarray:
    """Smooth margins for the three bounded state coordinates."""

    return np.concatenate(
        [
            parameters.rail_limit**2 - state[:, 0] ** 2,
            parameters.velocity_limit**2 - state[:, 1] ** 2,
            parameters.angular_velocity_limit**2 - state[:, 3] ** 2,
        ]
    )


def solve_single_shooting(
    scenario: SwingUpScenario = SwingUpScenario(),
    *,
    max_iterations: int = 700,
) -> TrajectoryResult:
    """Solve the swing-up program with controls as the only variables."""

    parameters = scenario.parameters
    guess = initial_control_guess(scenario)

    def objective(control: np.ndarray) -> float:
        state = rollout_controls(control, scenario)
        return swingup_objective(state, control, scenario)

    def state_constraints(control: np.ndarray) -> np.ndarray:
        return _state_constraint_margin(rollout_controls(control, scenario), parameters)

    result = minimize(
        objective,
        guess,
        method="SLSQP",
        bounds=[
            (-parameters.acceleration_limit, parameters.acceleration_limit)
            for _ in range(scenario.horizon_steps)
        ],
        constraints={"type": "ineq", "fun": state_constraints},
        options={"ftol": 1e-5, "maxiter": max_iterations},
    )
    state = rollout_controls(result.x, scenario)
    return TrajectoryResult(
        method="single shooting",
        time=np.arange(scenario.horizon_steps + 1) * scenario.step_size,
        state=state,
        control=np.asarray(result.x),
        objective=swingup_objective(state, result.x, scenario),
        success=bool(result.success),
        iterations=int(result.nit),
        dynamics_defect=0.0,
        message=str(result.message),
        decision_variables=scenario.horizon_steps,
        dynamics_equalities=0,
    )


def _step_jacobian(
    state: np.ndarray,
    control: float,
    scenario: SwingUpScenario,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically differentiate one RK4 step, preserving temporal sparsity."""

    state_matrix = np.empty((4, 4), dtype=float)
    for column in range(4):
        perturbation = np.zeros(4)
        perturbation[column] = epsilon
        forward = rk4_step(
            state + perturbation,
            control,
            scenario.parameters,
            scenario.step_size,
        )
        backward = rk4_step(
            state - perturbation,
            control,
            scenario.parameters,
            scenario.step_size,
        )
        state_matrix[:, column] = (forward - backward) / (2.0 * epsilon)
    input_matrix = (
        rk4_step(state, control + epsilon, scenario.parameters, scenario.step_size)
        - rk4_step(state, control - epsilon, scenario.parameters, scenario.step_size)
    )[:, None] / (2.0 * epsilon)
    return state_matrix, input_matrix


def solve_direct_transcription(
    scenario: SwingUpScenario = SwingUpScenario(),
    *,
    max_iterations: int = 700,
) -> TrajectoryResult:
    """Solve the matched program with states and controls as variables."""

    steps = scenario.horizon_steps
    state_count = 4 * (steps + 1)
    parameters = scenario.parameters
    initial_control = initial_control_guess(scenario)
    initial_state = rollout_controls(initial_control, scenario)
    initial_decision = np.concatenate([initial_state.ravel(), initial_control])

    def unpack(decision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return decision[:state_count].reshape(steps + 1, 4), decision[state_count:]

    def objective(decision: np.ndarray) -> float:
        state, control = unpack(decision)
        return swingup_objective(state, control, scenario)

    def objective_gradient(decision: np.ndarray) -> np.ndarray:
        state, control = unpack(decision)
        gradient_state = np.zeros_like(state)
        q_position, q_velocity, q_angle, q_angular_velocity = scenario.state_weights
        gradient_state[:-1, 0] = 2.0 * scenario.step_size * q_position * state[:-1, 0]
        gradient_state[:-1, 1] = 2.0 * scenario.step_size * q_velocity * state[:-1, 1]
        gradient_state[:-1, 2] = scenario.step_size * q_angle * np.sin(state[:-1, 2])
        gradient_state[:-1, 3] = (
            2.0 * scenario.step_size * q_angular_velocity * state[:-1, 3]
        )
        terminal = state[-1]
        f_position, f_velocity, f_angle, f_angular_velocity = scenario.terminal_weights
        gradient_state[-1, 0] = 2.0 * f_position * terminal[0]
        gradient_state[-1, 1] = 2.0 * f_velocity * terminal[1]
        gradient_state[-1, 2] = f_angle * np.sin(terminal[2])
        gradient_state[-1, 3] = 2.0 * f_angular_velocity * terminal[3]
        gradient_control = 2.0 * scenario.step_size * scenario.control_weight * control
        return np.concatenate([gradient_state.ravel(), gradient_control])

    def dynamics_residual(decision: np.ndarray) -> np.ndarray:
        state, control = unpack(decision)
        residual = np.empty((steps + 1, 4), dtype=float)
        residual[0] = state[0] - np.asarray(scenario.initial_state)
        for index in range(steps):
            residual[index + 1] = state[index + 1] - rk4_step(
                state[index],
                float(control[index]),
                parameters,
                scenario.step_size,
            )
        return residual.ravel()

    def dynamics_jacobian(decision: np.ndarray) -> np.ndarray:
        state, control = unpack(decision)
        dimension = state_count + steps
        jacobian = np.zeros((state_count, dimension), dtype=float)
        jacobian[:4, :4] = np.eye(4)
        for index in range(steps):
            state_matrix, input_matrix = _step_jacobian(
                state[index],
                float(control[index]),
                scenario,
            )
            row = 4 * (index + 1)
            jacobian[row : row + 4, 4 * index : 4 * (index + 1)] = -state_matrix
            jacobian[row : row + 4, 4 * (index + 1) : 4 * (index + 2)] = np.eye(4)
            jacobian[row : row + 4, state_count + index] = -input_matrix[:, 0]
        return jacobian

    lower_state = np.tile(
        [
            -parameters.rail_limit,
            -parameters.velocity_limit,
            -np.inf,
            -parameters.angular_velocity_limit,
        ],
        steps + 1,
    )
    upper_state = np.tile(
        [
            parameters.rail_limit,
            parameters.velocity_limit,
            np.inf,
            parameters.angular_velocity_limit,
        ],
        steps + 1,
    )
    bounds = Bounds(
        np.concatenate(
            [lower_state, np.full(steps, -parameters.acceleration_limit)]
        ),
        np.concatenate(
            [upper_state, np.full(steps, parameters.acceleration_limit)]
        ),
    )
    result = minimize(
        objective,
        initial_decision,
        jac=objective_gradient,
        method="SLSQP",
        bounds=bounds,
        constraints={
            "type": "eq",
            "fun": dynamics_residual,
            "jac": dynamics_jacobian,
        },
        options={"ftol": 1e-5, "maxiter": max_iterations},
    )
    state, control = unpack(result.x)
    defect = float(np.max(np.abs(dynamics_residual(result.x))))
    return TrajectoryResult(
        method="direct transcription",
        time=np.arange(steps + 1) * scenario.step_size,
        state=np.asarray(state),
        control=np.asarray(control),
        objective=swingup_objective(state, control, scenario),
        success=bool(result.success),
        iterations=int(result.nit),
        dynamics_defect=defect,
        message=str(result.message),
        decision_variables=state_count + steps,
        dynamics_equalities=state_count,
    )


@lru_cache(maxsize=2)
def solve_swingup_comparison(
    scenario: SwingUpScenario = SwingUpScenario(),
) -> Mapping[str, TrajectoryResult]:
    """Run both formulations from the same control initialization."""

    return {
        "direct": solve_direct_transcription(scenario),
        "shooting": solve_single_shooting(scenario),
    }


def format_swingup_metrics(results: Mapping[str, TrajectoryResult]) -> str:
    """Format the compact solver comparison printed beside the figure."""

    lines = [
        "method                variables  eqs  iterations  objective  final angle  defect"
    ]
    for key in ("direct", "shooting"):
        result = results[key]
        final_angle = abs(float(np.rad2deg(wrap_angle(result.state[-1, 2]))))
        lines.append(
            f"{result.method:<21} {result.decision_variables:>5d} "
            f"{result.dynamics_equalities:>4d} {result.iterations:>11d} "
            f"{result.objective:>10.3f} {final_angle:>10.3f} deg "
            f"{result.dynamics_defect:>8.1e}"
        )
    return "\n".join(lines)


def make_swingup_figure(
    results: Mapping[str, TrajectoryResult],
    scenario: SwingUpScenario = SwingUpScenario(),
) -> plt.Figure:
    """Compare the two swing-up solutions on shared physical axes."""

    specifications = (
        (results["direct"], DIRECT_COLOR, "-", "direct transcription"),
        (results["shooting"], SHOOTING_COLOR, "--", "single shooting"),
    )
    with mpl.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
        for result, color, style, label in specifications:
            axes[0].plot(result.time, result.state[:, 0], color=color, linestyle=style, label=label)
            axes[1].plot(result.time, np.cos(result.state[:, 2]), color=color, linestyle=style)
            axes[2].step(
                result.time[:-1],
                result.control,
                where="post",
                color=color,
                linestyle=style,
            )

        limit = scenario.parameters.rail_limit
        axes[0].axhline(limit, color="0.65", linestyle=":", linewidth=0.8)
        axes[0].axhline(-limit, color="0.65", linestyle=":", linewidth=0.8)
        axes[0].set(xlabel="time (s)", ylabel="cart position (m)")
        axes[0].legend(loc="best")

        axes[1].axhline(1.0, color="0.65", linestyle=":", linewidth=0.8)
        axes[1].set(
            xlabel="time (s)",
            ylabel=r"normalized pole height $\cos\theta$",
            ylim=(-1.08, 1.08),
        )
        axes[1].annotate(
            "upright",
            xy=(scenario.duration, 1.0),
            xytext=(-3, -12),
            textcoords="offset points",
            ha="right",
            color="0.35",
        )

        acceleration_limit = scenario.parameters.acceleration_limit
        axes[2].axhline(acceleration_limit, color="0.65", linestyle=":", linewidth=0.8)
        axes[2].axhline(-acceleration_limit, color="0.65", linestyle=":", linewidth=0.8)
        axes[2].set(xlabel="time (s)", ylabel=r"cart acceleration (m s$^{-2}$)")

        for axis in axes:
            axis.grid(axis="y", color="0.91", linewidth=0.55)
        return figure


def _draw_cartpole(
    axis: plt.Axes,
    state: np.ndarray,
    parameters: CartPoleParameters,
    color: str,
) -> tuple[Rectangle, plt.Line2D, Circle]:
    position, _, angle, _ = state
    cart = Rectangle(
        (position - 0.18, -0.08),
        0.36,
        0.16,
        facecolor="0.88",
        edgecolor="0.25",
        linewidth=0.8,
    )
    axis.add_patch(cart)
    pivot_y = 0.08
    bob_x = position + parameters.pole_length * np.sin(angle)
    bob_y = pivot_y + parameters.pole_length * np.cos(angle)
    pole, = axis.plot([position, bob_x], [pivot_y, bob_y], color=color, linewidth=2.0)
    bob = Circle((bob_x, bob_y), radius=0.045, facecolor=color, edgecolor="none")
    axis.add_patch(bob)
    return cart, pole, bob


def make_swingup_animation(
    results: Mapping[str, TrajectoryResult],
    scenario: SwingUpScenario = SwingUpScenario(),
    *,
    fps: int = 12,
) -> FuncAnimation:
    """Animate both optimized trajectories with identical geometry and timing."""

    with mpl.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
        entries = (
            (results["direct"], DIRECT_COLOR, "direct transcription"),
            (results["shooting"], SHOOTING_COLOR, "single shooting"),
        )
        artists = []
        for axis, (result, color, title) in zip(axes, entries):
            axis.axhline(0.0, color="0.45", linewidth=0.8)
            axis.axvline(-scenario.parameters.rail_limit, color="0.75", linestyle=":", linewidth=0.8)
            axis.axvline(scenario.parameters.rail_limit, color="0.75", linestyle=":", linewidth=0.8)
            axis.set(
                xlim=(-scenario.parameters.rail_limit - 0.35, scenario.parameters.rail_limit + 0.35),
                ylim=(-0.60, 0.78),
                aspect="equal",
                title=title,
                xlabel="cart position (m)",
            )
            cart, pole, bob = _draw_cartpole(axis, result.state[0], scenario.parameters, color)
            time_label = axis.text(0.03, 0.92, "", transform=axis.transAxes, color="0.25")
            artists.append((cart, pole, bob, time_label, result, color))

        def update(frame: int) -> list[object]:
            changed: list[object] = []
            for cart, pole, bob, time_label, result, color in artists:
                position, _, angle, _ = result.state[frame]
                cart.set_x(position - 0.18)
                pivot_y = 0.08
                bob_x = position + scenario.parameters.pole_length * np.sin(angle)
                bob_y = pivot_y + scenario.parameters.pole_length * np.cos(angle)
                pole.set_data([position, bob_x], [pivot_y, bob_y])
                pole.set_color(color)
                bob.center = (bob_x, bob_y)
                time_label.set_text(f"t = {result.time[frame]:.2f} s")
                changed.extend((cart, pole, bob, time_label))
            return changed

        animation = FuncAnimation(
            figure,
            update,
            frames=scenario.horizon_steps + 1,
            interval=1000.0 / fps,
            blit=False,
        )
        return animation


def replay_open_loop_with_disturbance(
    result: TrajectoryResult,
    scenario: SwingUpScenario = SwingUpScenario(),
    *,
    disturbance_step: int = 14,
    disturbance_acceleration: float = 1.0,
) -> OpenLoopReplay:
    """Replay one plan after an unmodeled one-step acceleration impulse."""

    if not 0 <= disturbance_step < scenario.horizon_steps:
        raise ValueError("disturbance_step must lie inside the planning horizon")
    nominal_state = rollout_controls(result.control, scenario)
    disturbed_state = np.empty_like(nominal_state)
    disturbed_state[0] = np.asarray(scenario.initial_state)
    disturbance = np.zeros(scenario.horizon_steps)
    disturbance[disturbance_step] = disturbance_acceleration
    for index, command in enumerate(result.control):
        disturbed_state[index + 1] = rk4_step(
            disturbed_state[index],
            float(command + disturbance[index]),
            scenario.parameters,
            scenario.step_size,
        )
    return OpenLoopReplay(
        time=np.arange(scenario.horizon_steps + 1) * scenario.step_size,
        nominal_state=nominal_state,
        disturbed_state=disturbed_state,
        command=result.control.copy(),
        disturbance=disturbance,
    )


def make_open_loop_perturbation_figure(
    replay: OpenLoopReplay,
) -> plt.Figure:
    """Show the divergence caused by a disturbance the fixed plan cannot observe."""

    with mpl.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.35), constrained_layout=True)
        specifications = (
            (replay.nominal_state, DIRECT_COLOR, "-", "nominal realization"),
            (replay.disturbed_state, SHOOTING_COLOR, "--", "one-step disturbance"),
        )
        for state, color, style, label in specifications:
            axes[0].plot(
                replay.time,
                np.cos(state[:, 2]),
                color=color,
                linestyle=style,
                label=label,
            )
            axes[1].plot(
                replay.time,
                state[:, 0],
                color=color,
                linestyle=style,
            )
        disturbance_index = int(np.flatnonzero(replay.disturbance)[0])
        disturbance_time = replay.time[disturbance_index]
        for axis in axes:
            axis.axvline(disturbance_time, color="0.45", linestyle=":", linewidth=0.8)
            axis.grid(axis="y", color="0.91", linewidth=0.55)
        axes[0].set(
            xlabel="time (s)",
            ylabel=r"normalized pole height $\cos\theta$",
            ylim=(-1.08, 1.08),
        )
        axes[0].legend(loc="lower left")
        axes[0].annotate(
            r"$1\;\mathrm{m\,s^{-2}}$ for one step",
            xy=(disturbance_time, 0.0),
            xytext=(7, 8),
            textcoords="offset points",
            color="0.28",
        )
        axes[1].set(xlabel="time (s)", ylabel="cart position (m)")
        return figure


def linearize_upright(
    parameters: CartPoleParameters = CartPoleParameters(),
    *,
    step_size: float = 0.02,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearize the exact discrete RK4 update at the upright equilibrium."""

    equilibrium = np.zeros(4)
    state_matrix = np.empty((4, 4), dtype=float)
    for column in range(4):
        perturbation = np.zeros(4)
        perturbation[column] = epsilon
        state_matrix[:, column] = (
            rk4_step(equilibrium + perturbation, 0.0, parameters, step_size)
            - rk4_step(equilibrium - perturbation, 0.0, parameters, step_size)
        ) / (2.0 * epsilon)
    input_matrix = (
        rk4_step(equilibrium, epsilon, parameters, step_size)
        - rk4_step(equilibrium, -epsilon, parameters, step_size)
    )[:, None] / (2.0 * epsilon)
    return state_matrix, input_matrix


def design_lqr(
    parameters: CartPoleParameters = CartPoleParameters(),
    *,
    step_size: float = 0.02,
) -> LQRDesign:
    """Solve the discrete algebraic Riccati equation at the upright state."""

    state_matrix, input_matrix = linearize_upright(parameters, step_size=step_size)
    cost_matrix = np.diag([2.0, 0.2, 80.0, 3.0])
    control_cost = np.array([[0.15]])
    riccati_matrix = solve_discrete_are(
        state_matrix,
        input_matrix,
        cost_matrix,
        control_cost,
    )
    gain = np.linalg.solve(
        control_cost + input_matrix.T @ riccati_matrix @ input_matrix,
        input_matrix.T @ riccati_matrix @ state_matrix,
    )
    eigenvalues = np.linalg.eigvals(state_matrix - input_matrix @ gain)
    return LQRDesign(
        state_matrix,
        input_matrix,
        cost_matrix,
        control_cost,
        riccati_matrix,
        gain,
        eigenvalues,
    )


def simulate_lqr(
    initial_angle_deg: float,
    *,
    controlled: bool,
    parameters: CartPoleParameters = CartPoleParameters(),
    design: LQRDesign | None = None,
    step_size: float = 0.02,
    duration: float = 6.0,
) -> ClosedLoopResult:
    """Validate the linear controller on the constrained nonlinear plant."""

    if controlled and design is None:
        design = design_lqr(parameters, step_size=step_size)
    state_values = [np.array([0.0, 0.0, np.deg2rad(initial_angle_deg), 0.0])]
    control_values: list[float] = []
    terminated = False
    reason: str | None = None
    steps = int(round(duration / step_size))

    for _ in range(steps):
        state = state_values[-1]
        if controlled:
            error = state.copy()
            error[2] = wrap_angle(error[2])
            acceleration = -float((design.gain @ error).item())  # type: ignore[union-attr]
            acceleration = float(
                np.clip(
                    acceleration,
                    -parameters.acceleration_limit,
                    parameters.acceleration_limit,
                )
            )
        else:
            acceleration = 0.0

        next_state = rk4_step(state, acceleration, parameters, step_size)
        if abs(next_state[0]) >= parameters.rail_limit:
            denominator = next_state[0] - state[0]
            if abs(denominator) > 1e-12:
                fraction = (
                    np.sign(next_state[0]) * parameters.rail_limit - state[0]
                ) / denominator
                next_state = state + np.clip(fraction, 0.0, 1.0) * (next_state - state)
            next_state[0] = np.sign(next_state[0]) * parameters.rail_limit
            terminated = True
            reason = "rail limit"

        state_values.append(next_state)
        control_values.append(acceleration)
        if terminated:
            break

    name = ("LQR" if controlled else "uncontrolled") + f", {initial_angle_deg:g} deg"
    state_array = np.asarray(state_values)
    return ClosedLoopResult(
        name=name,
        initial_angle_deg=initial_angle_deg,
        time=np.arange(len(state_array)) * step_size,
        state=state_array,
        control=np.asarray(control_values),
        terminated=terminated,
        termination_reason=reason,
    )


@lru_cache(maxsize=2)
def run_lqr_cases(
    parameters: CartPoleParameters = CartPoleParameters(),
) -> Mapping[str, ClosedLoopResult]:
    """Run the matched uncontrolled, local-success, and large-angle cases."""

    design = design_lqr(parameters)
    return {
        "uncontrolled": simulate_lqr(5.0, controlled=False, parameters=parameters),
        "local": simulate_lqr(5.0, controlled=True, parameters=parameters, design=design),
        "large": simulate_lqr(45.0, controlled=True, parameters=parameters, design=design),
    }


def format_lqr_metrics(
    cases: Mapping[str, ClosedLoopResult],
    design: LQRDesign,
) -> str:
    """Return stability and nonlinear rollout diagnostics for the text."""

    lines = [f"closed-loop spectral radius: {max(abs(design.closed_loop_eigenvalues)):.4f}"]
    for key in ("uncontrolled", "local", "large"):
        result = cases[key]
        status = result.termination_reason or ("stabilized" if result.stabilized else "not stabilized")
        peak_control = float(np.max(np.abs(result.control))) if result.control.size else 0.0
        lines.append(
            f"{result.name:<22} {status:<14} "
            f"final angle={result.final_angle_error_deg:>7.2f} deg "
            f"max |x|={np.max(np.abs(result.state[:, 0])):>5.2f} m "
            f"max |u|={peak_control:>4.1f} m/s^2"
        )
    return "\n".join(lines)


def make_lqr_figure(
    cases: Mapping[str, ClosedLoopResult],
    parameters: CartPoleParameters = CartPoleParameters(),
) -> plt.Figure:
    """Plot local recovery and the nonlinear constraint-induced failure."""

    specifications = (
        (cases["uncontrolled"], UNCONTROLLED_COLOR, ":"),
        (cases["local"], DIRECT_COLOR, "-"),
        (cases["large"], SHOOTING_COLOR, "--"),
    )
    with mpl.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
        for result, color, style in specifications:
            label = result.name
            axes[0].plot(
                result.time,
                np.rad2deg(wrap_angle(result.state[:, 2])),
                color=color,
                linestyle=style,
                label=label,
            )
            axes[1].plot(result.time, result.state[:, 0], color=color, linestyle=style)
            axes[2].step(
                result.time[:-1],
                result.control,
                where="post",
                color=color,
                linestyle=style,
            )

        axes[0].axhline(0.0, color="0.65", linewidth=0.8)
        axes[0].set(xlabel="time (s)", ylabel="angle from upright (deg)")
        axes[0].legend(loc="best")

        axes[1].axhline(parameters.rail_limit, color="0.65", linestyle=":", linewidth=0.8)
        axes[1].axhline(-parameters.rail_limit, color="0.65", linestyle=":", linewidth=0.8)
        axes[1].set(xlabel="time (s)", ylabel="cart position (m)")
        large = cases["large"]
        axes[1].annotate(
            "rail limit",
            xy=(large.time[-1], large.state[-1, 0]),
            xytext=(-18, -20),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "0.35", "linewidth": 0.8},
            ha="right",
            color="0.25",
        )

        axes[2].axhline(parameters.acceleration_limit, color="0.65", linestyle=":", linewidth=0.8)
        axes[2].axhline(-parameters.acceleration_limit, color="0.65", linestyle=":", linewidth=0.8)
        axes[2].set(xlabel="time (s)", ylabel=r"cart acceleration (m s$^{-2}$)")
        axes[2].annotate(
            "saturation",
            xy=(large.time[1], large.control[0]),
            xytext=(18, 14),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "0.35", "linewidth": 0.8},
            color="0.25",
        )

        for axis in axes:
            axis.grid(axis="y", color="0.91", linewidth=0.55)
        return figure


def make_lqr_animation(
    cases: Mapping[str, ClosedLoopResult],
    parameters: CartPoleParameters = CartPoleParameters(),
    *,
    stride: int = 4,
    fps: int = 16,
) -> FuncAnimation:
    """Animate the uncontrolled, local-success, and large-angle rollouts."""

    with mpl.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
        entries = (
            (cases["uncontrolled"], UNCONTROLLED_COLOR),
            (cases["local"], DIRECT_COLOR),
            (cases["large"], SHOOTING_COLOR),
        )
        artists = []
        for axis, (result, color) in zip(axes, entries):
            axis.axhline(0.0, color="0.45", linewidth=0.8)
            axis.axvline(-parameters.rail_limit, color="0.75", linestyle=":", linewidth=0.8)
            axis.axvline(parameters.rail_limit, color="0.75", linestyle=":", linewidth=0.8)
            axis.set(
                xlim=(-parameters.rail_limit - 0.35, parameters.rail_limit + 0.35),
                ylim=(-0.60, 0.78),
                aspect="equal",
                title=result.name,
                xlabel="cart position (m)",
            )
            cart, pole, bob = _draw_cartpole(axis, result.state[0], parameters, color)
            time_label = axis.text(0.03, 0.92, "", transform=axis.transAxes, color="0.25")
            status_label = axis.text(
                0.03,
                0.83,
                "",
                transform=axis.transAxes,
                color=SHOOTING_COLOR,
            )
            artists.append((cart, pole, bob, time_label, status_label, result, color))

        frame_indices = np.arange(0, max(len(item[5].state) for item in artists), stride)
        if frame_indices[-1] != max(len(item[5].state) for item in artists) - 1:
            frame_indices = np.append(frame_indices, max(len(item[5].state) for item in artists) - 1)

        def update(frame: int) -> list[object]:
            changed: list[object] = []
            index_requested = int(frame_indices[frame])
            for cart, pole, bob, time_label, status_label, result, color in artists:
                index = min(index_requested, len(result.state) - 1)
                position, _, angle, _ = result.state[index]
                cart.set_x(position - 0.18)
                pivot_y = 0.08
                bob_x = position + parameters.pole_length * np.sin(angle)
                bob_y = pivot_y + parameters.pole_length * np.cos(angle)
                pole.set_data([position, bob_x], [pivot_y, bob_y])
                bob.center = (bob_x, bob_y)
                time_label.set_text(f"t = {result.time[index]:.2f} s")
                if result.terminated and index == len(result.state) - 1:
                    status_label.set_text(result.termination_reason or "terminated")
                changed.extend((cart, pole, bob, time_label, status_label))
            return changed

        animation = FuncAnimation(
            figure,
            update,
            frames=len(frame_indices),
            interval=1000.0 / fps,
            blit=False,
        )
        return animation


def _smoke_test() -> None:
    comparison = solve_swingup_comparison()
    if not all(result.success for result in comparison.values()):
        raise AssertionError("both swing-up solvers should terminate successfully")
    for result in comparison.values():
        if np.cos(result.state[-1, 2]) < 0.99:
            raise AssertionError(f"{result.method} did not reach upright")
        if result.dynamics_defect > 1e-4:
            raise AssertionError(f"{result.method} has a large dynamics defect")

    design = design_lqr()
    if np.max(np.abs(design.closed_loop_eigenvalues)) >= 1.0:
        raise AssertionError("the linear closed loop should be asymptotically stable")
    cases = run_lqr_cases()
    if not cases["local"].stabilized:
        raise AssertionError("the local LQR case should stabilize")
    if not cases["large"].terminated:
        raise AssertionError("the large-angle case should reach the rail limit")

    swing_figure = make_swingup_figure(comparison)
    lqr_figure = make_lqr_figure(cases)
    plt.close(swing_figure)
    plt.close(lqr_figure)
    print(format_swingup_metrics(comparison))
    print(format_lqr_metrics(cases, design))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", type=Path, help="save the two static figures")
    arguments = parser.parse_args()
    _smoke_test()
    if arguments.save_dir is not None:
        arguments.save_dir.mkdir(parents=True, exist_ok=True)
        make_swingup_figure(solve_swingup_comparison()).savefig(
            arguments.save_dir / "cartpole_swingup.pdf"
        )
        make_lqr_figure(run_lqr_cases()).savefig(arguments.save_dir / "cartpole_lqr.pdf")
