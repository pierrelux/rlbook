"""A small controlled BIXI inventory model for the modeling chapter.

The module keeps three objects separate:

* completed-trip data used to calibrate an event-rate model;
* a counterfactual station-and-truck simulator;
* controllers that receive current inventories but not future events.

The public BIXI archive contains completed trips, not attempted demand or
operator relocation.  Results from this module therefore describe the teaching
model.  They are not estimates of service failures in the operated BIXI system.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Literal, Protocol

import numpy as np


EventKind = Literal["rental", "return"]
ControllerName = Literal["none", "open_loop", "feedback"]


@dataclass(frozen=True)
class Station:
    station_id: str
    external_id: str
    short_name: str
    name: str
    latitude: float
    longitude: float
    capacity: int

    def validate(self) -> None:
        if not self.station_id or not self.name:
            raise ValueError("station identifiers and names must be nonempty")
        if self.capacity <= 0:
            raise ValueError("station capacity must be positive")
        if not (-90.0 <= self.latitude <= 90.0):
            raise ValueError("station latitude is invalid")
        if not (-180.0 <= self.longitude <= 180.0):
            raise ValueError("station longitude is invalid")


@dataclass(frozen=True)
class BixiScenario:
    stations: tuple[Station, ...]
    horizon_start_local: str
    horizon_minutes: int
    control_period_minutes: int
    initial_inventory: np.ndarray
    truck_capacity: int
    transfer_limit: int
    initial_truck_inventory: int
    initial_truck_station: int
    travel_steps: np.ndarray
    distance_km: np.ndarray
    failure_weight: float
    bike_moved_weight: float
    distance_weight: float
    terminal_imbalance_weight: float

    @property
    def capacities(self) -> np.ndarray:
        return np.asarray([station.capacity for station in self.stations], dtype=float)

    @property
    def decision_count(self) -> int:
        return self.horizon_minutes // self.control_period_minutes

    def validate(self) -> None:
        if len(self.stations) != 3:
            raise ValueError("the textbook scenario must contain exactly three stations")
        for station in self.stations:
            station.validate()
        if self.horizon_minutes <= 0 or self.control_period_minutes <= 0:
            raise ValueError("time settings must be positive")
        if self.horizon_minutes % self.control_period_minutes:
            raise ValueError("the horizon must contain a whole number of control periods")
        inventory = np.asarray(self.initial_inventory, dtype=float)
        if inventory.shape != (len(self.stations),):
            raise ValueError("initial_inventory has the wrong shape")
        if np.any(inventory < 0.0) or np.any(inventory > self.capacities):
            raise ValueError("initial station inventory violates capacity bounds")
        if self.truck_capacity <= 0 or self.transfer_limit <= 0:
            raise ValueError("truck limits must be positive")
        if not 0 <= self.initial_truck_inventory <= self.truck_capacity:
            raise ValueError("initial truck inventory is infeasible")
        if not 0 <= self.initial_truck_station < len(self.stations):
            raise ValueError("initial truck station is invalid")
        shape = (len(self.stations), len(self.stations))
        travel = np.asarray(self.travel_steps)
        distance = np.asarray(self.distance_km, dtype=float)
        if travel.shape != shape or distance.shape != shape:
            raise ValueError("travel matrices have the wrong shape")
        if np.any(travel < 0) or np.any(distance < 0.0):
            raise ValueError("travel times and distances must be nonnegative")
        if np.any(np.diag(travel) != 0) or np.any(np.diag(distance) != 0.0):
            raise ValueError("travel-matrix diagonals must be zero")
        if not np.array_equal(travel, travel.T):
            raise ValueError("travel_steps must be symmetric")
        if not np.allclose(distance, distance.T):
            raise ValueError("distance_km must be symmetric")
        if np.any(travel[~np.eye(len(self.stations), dtype=bool)] != 1):
            raise ValueError("this reduced model requires one-period interstation travel")
        weights = np.asarray(
            [
                self.failure_weight,
                self.bike_moved_weight,
                self.distance_weight,
                self.terminal_imbalance_weight,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("objective weights must be finite and nonnegative")
        if self.failure_weight <= 0.0:
            raise ValueError("the service-failure weight must be positive")


@dataclass(frozen=True)
class CompletedEventProfile:
    bin_start_minutes: np.ndarray
    completed_starts: np.ndarray
    completed_ends: np.ndarray
    training_day_count: int
    source_label: str

    @property
    def decision_count(self) -> int:
        return int(self.bin_start_minutes.size)

    def validate(self, station_count: int) -> None:
        starts = np.asarray(self.completed_starts, dtype=float)
        ends = np.asarray(self.completed_ends, dtype=float)
        if starts.shape != ends.shape:
            raise ValueError("start and end rate arrays must have the same shape")
        if starts.shape != (self.decision_count, station_count):
            raise ValueError("completed-event rate arrays have the wrong shape")
        if np.any(starts < 0.0) or np.any(ends < 0.0):
            raise ValueError("completed-event rates must be nonnegative")
        if np.any(~np.isfinite(starts)) or np.any(~np.isfinite(ends)):
            raise ValueError("completed-event rates must be finite")
        if self.training_day_count <= 0:
            raise ValueError("training_day_count must be positive")


@dataclass(frozen=True, order=True)
class Event:
    time_minutes: float
    station_index: int
    kind: EventKind
    source: str = ""

    def validate(self, scenario: BixiScenario) -> None:
        if not np.isfinite(self.time_minutes):
            raise ValueError("event time must be finite")
        if not 0.0 <= self.time_minutes < scenario.horizon_minutes:
            raise ValueError("event lies outside the simulation horizon")
        if not 0 <= self.station_index < len(scenario.stations):
            raise ValueError("event station is invalid")
        if self.kind not in ("rental", "return"):
            raise ValueError("event kind must be rental or return")


@dataclass(frozen=True)
class EventTrace:
    events: tuple[Event, ...]
    source_label: str
    seed: int | None = None
    includes_paired_pulse: bool = False

    def validate(self, scenario: BixiScenario) -> None:
        previous = -np.inf
        for event in self.events:
            event.validate(scenario)
            if event.time_minutes < previous:
                raise ValueError("events must be sorted by time")
            previous = event.time_minutes


@dataclass(frozen=True)
class BixiObservation:
    decision_index: int
    time_minutes: float
    station_inventory: np.ndarray
    truck_inventory: int
    truck_station: int


@dataclass(frozen=True)
class BixiAction:
    transfer: int
    destination: int


class Controller(Protocol):
    name: ControllerName

    def __call__(self, observation: BixiObservation) -> BixiAction: ...


@dataclass(frozen=True)
class BixiMetrics:
    attempted_rentals: int
    served_rentals: int
    attempted_returns: int
    accepted_returns: int
    lost_rentals: int
    rejected_returns: int
    service_failures: int
    bikes_moved: int
    truck_distance_km: float
    unrealized_transfer_bikes: int
    terminal_imbalance: float
    terminal_truck_inventory: int
    objective: float


@dataclass(frozen=True)
class BixiTrajectory:
    controller_name: ControllerName
    source_label: str
    checkpoint_time_minutes: np.ndarray
    station_inventory: np.ndarray
    truck_inventory: np.ndarray
    truck_station: np.ndarray
    requested_transfer: np.ndarray
    realized_transfer: np.ndarray
    destination: np.ndarray
    cumulative_lost_rentals: np.ndarray
    cumulative_rejected_returns: np.ndarray
    cumulative_bikes_moved: np.ndarray
    cumulative_distance_km: np.ndarray
    event_time_minutes: np.ndarray
    event_station: np.ndarray
    event_kind: tuple[EventKind, ...]
    event_accepted: np.ndarray
    event_inventory_after: np.ndarray
    history_time_minutes: np.ndarray
    history_station_inventory: np.ndarray
    history_truck_inventory: np.ndarray
    metrics: BixiMetrics

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                payload[key] = value.tolist()
            elif isinstance(value, dict):
                payload[key] = value
            else:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class FluidTrajectory:
    checkpoint_time_minutes: np.ndarray
    station_inventory: np.ndarray
    cumulative_lost_rentals: np.ndarray
    cumulative_rejected_returns: np.ndarray


class NoRelocationController:
    name: ControllerName = "none"

    def __call__(self, observation: BixiObservation) -> BixiAction:
        return BixiAction(transfer=0, destination=observation.truck_station)


class FrozenScheduleController:
    name: ControllerName = "open_loop"

    def __init__(self, actions: Sequence[BixiAction]):
        self.actions = tuple(actions)

    def __call__(self, observation: BixiObservation) -> BixiAction:
        return self.actions[observation.decision_index]


class InventoryFeedbackController:
    name: ControllerName = "feedback"

    def __init__(self, scenario: BixiScenario, profile: CompletedEventProfile):
        scenario.validate()
        profile.validate(len(scenario.stations))
        if profile.decision_count != scenario.decision_count:
            raise ValueError("profile and scenario horizons differ")
        self.scenario = scenario
        self.profile = profile

    def target_inventory(self, decision_index: int) -> np.ndarray:
        remaining_net_rentals = np.sum(
            self.profile.completed_starts[decision_index:]
            - self.profile.completed_ends[decision_index:],
            axis=0,
        )
        capacities = self.scenario.capacities
        return np.clip(
            0.5 * capacities + remaining_net_rentals,
            0.2 * capacities,
            0.8 * capacities,
        )

    def __call__(self, observation: BixiObservation) -> BixiAction:
        scenario = self.scenario
        x = np.asarray(observation.station_inventory, dtype=float)
        b = int(observation.truck_inventory)
        location = int(observation.truck_station)
        target = self.target_inventory(observation.decision_index)
        deficit = np.maximum(target - x, 0.0)
        surplus = np.maximum(x - target, 0.0)

        transfer = 0
        if b > 0 and deficit[location] >= 2.0:
            transfer = int(
                min(
                    scenario.transfer_limit,
                    b,
                    np.ceil(deficit[location]),
                    scenario.capacities[location] - x[location],
                )
            )
        else:
            remote = np.arange(len(scenario.stations)) != location
            remote_deficit = float(np.sum(deficit[remote]))
            if (
                surplus[location] >= 2.0
                and b < scenario.truck_capacity
                and remote_deficit >= 2.0
            ):
                load = int(
                    min(
                        scenario.transfer_limit,
                        scenario.truck_capacity - b,
                        np.floor(surplus[location]),
                        np.ceil(remote_deficit),
                        x[location],
                    )
                )
                transfer = -load
            elif (
                b > 0
                and float(np.sum(deficit)) < 2.0
                and scenario.capacities[location] - x[location] >= 1.0
            ):
                # Bikes left on the truck are unavailable to customers.  Return
                # them when no modeled target deficit remains.
                transfer = int(
                    min(
                        scenario.transfer_limit,
                        b,
                        scenario.capacities[location] - x[location],
                    )
                )

        x_after = x.copy()
        x_after[location] += transfer
        b_after = b - transfer
        deficit_after = np.maximum(target - x_after, 0.0)
        surplus_after = np.maximum(x_after - target, 0.0)

        if b_after > 0 and float(np.max(deficit_after)) >= 2.0:
            score = deficit_after / scenario.capacities
        elif (
            b_after == 0
            and float(np.max(deficit_after)) >= 2.0
            and float(np.max(surplus_after)) >= 2.0
        ):
            score = surplus_after / scenario.capacities
        else:
            return BixiAction(transfer=transfer, destination=location)

        score = score.copy()
        score[location] = -np.inf
        best = float(np.max(score))
        if best <= 0.0:
            destination = location
        else:
            tied = np.flatnonzero(np.isclose(score, best))
            destination = min(
                (int(index) for index in tied),
                key=lambda index: (
                    scenario.distance_km[location, index],
                    index,
                ),
            )
        return BixiAction(transfer=transfer, destination=destination)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_scenario(path: Path | str) -> BixiScenario:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = payload["stations"]
    stations = tuple(
        Station(
            station_id=str(record["station_id"]),
            external_id=str(record["external_id"]),
            short_name=str(record["short_name"]),
            name=str(record["name"]),
            latitude=float(record["lat"]),
            longitude=float(record["lon"]),
            capacity=int(record["capacity"]),
        )
        for record in records
    )
    scenario_payload = payload["scenario"]
    objective = scenario_payload["objective"]
    scenario = BixiScenario(
        stations=stations,
        horizon_start_local=str(scenario_payload["horizon_start_local"]),
        horizon_minutes=int(scenario_payload["horizon_minutes"]),
        control_period_minutes=int(scenario_payload["control_period_minutes"]),
        initial_inventory=np.asarray(scenario_payload["initial_inventory"], dtype=float),
        truck_capacity=int(scenario_payload["truck_capacity"]),
        transfer_limit=int(scenario_payload["transfer_limit"]),
        initial_truck_inventory=int(scenario_payload["initial_truck_inventory"]),
        initial_truck_station=int(scenario_payload["initial_truck_station"]),
        travel_steps=np.asarray(scenario_payload["travel_steps"], dtype=int),
        distance_km=np.asarray(scenario_payload["distance_km"], dtype=float),
        failure_weight=float(objective["service_failure"]),
        bike_moved_weight=float(objective["bike_moved"]),
        distance_weight=float(objective["truck_distance_km"]),
        terminal_imbalance_weight=float(objective["terminal_imbalance"]),
    )
    scenario.validate()
    return scenario


def load_completed_profile(
    path: Path | str,
    scenario: BixiScenario,
) -> CompletedEventProfile:
    source = Path(path)
    by_bin: dict[int, dict[str, tuple[float, float, int]]] = {}
    with source.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            start = int(row["bin_start_minutes"])
            by_bin.setdefault(start, {})[row["station_name"]] = (
                float(row["mean_completed_starts"]),
                float(row["mean_completed_ends"]),
                int(row["training_day_count"]),
            )
    starts: list[list[float]] = []
    ends: list[list[float]] = []
    training_days: set[int] = set()
    bins = sorted(by_bin)
    for start in bins:
        starts.append([])
        ends.append([])
        for station in scenario.stations:
            record = by_bin[start].get(station.name)
            if record is None:
                raise ValueError(f"missing profile row for {station.name!r} at {start}")
            starts[-1].append(record[0])
            ends[-1].append(record[1])
            training_days.add(record[2])
    if len(training_days) != 1:
        raise ValueError("profile rows disagree about the training-day count")
    profile = CompletedEventProfile(
        bin_start_minutes=np.asarray(bins, dtype=float),
        completed_starts=np.asarray(starts, dtype=float),
        completed_ends=np.asarray(ends, dtype=float),
        training_day_count=training_days.pop(),
        source_label="BIXI 2024 completed trips, 43 training weekdays",
    )
    profile.validate(len(scenario.stations))
    if profile.decision_count != scenario.decision_count:
        raise ValueError("profile must contain one row per control period")
    return profile


def load_event_trace(path: Path | str, scenario: BixiScenario) -> EventTrace:
    source = Path(path)
    name_to_index = {station.name: index for index, station in enumerate(scenario.stations)}
    events: list[Event] = []
    with source.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            events.append(
                Event(
                    time_minutes=float(row["time_minutes"]),
                    station_index=name_to_index[row["station_name"]],
                    kind=str(row["kind"]),  # type: ignore[arg-type]
                    source="BIXI completed trip",
                )
            )
    kind_order = {"return": 0, "rental": 1}
    events.sort(
        key=lambda event: (
            event.time_minutes,
            event.station_index,
            kind_order[event.kind],
        )
    )
    trace = EventTrace(
        events=tuple(events),
        source_label="BIXI completed-trip timestamps, 2024-07-04 07:00-10:00",
    )
    trace.validate(scenario)
    return trace


def sample_poisson_trace(
    scenario: BixiScenario,
    profile: CompletedEventProfile,
    seed: int,
    *,
    paired_pulse: bool = False,
) -> EventTrace:
    """Sample events from the transparent completed-event-rate model.

    With ``paired_pulse=True``, eight additional rental attempts occur at
    Prince-Arthur at 07:47 and eight independent return attempts occur at
    Aylmer at 07:57.  The disturbance is identical across controllers.  Event
    acceptance still depends on the controlled inventory.
    """

    profile.validate(len(scenario.stations))
    rng = np.random.default_rng(seed)
    events: list[Event] = []
    period = float(scenario.control_period_minutes)
    for decision in range(scenario.decision_count):
        start_time = decision * period
        for station in range(len(scenario.stations)):
            for kind, rate in (
                ("rental", profile.completed_starts[decision, station]),
                ("return", profile.completed_ends[decision, station]),
            ):
                count = int(rng.poisson(rate))
                times = start_time + period * rng.random(count)
                events.extend(
                    Event(float(time), station, kind, "Poisson completed-event model")
                    for time in times
                )
    if paired_pulse:
        # The pulses occur after the 07:45 action and before the 08:00 action
        # decision.  They are independent potential events, not a guaranteed
        # transfer of eight physical bicycles.
        events.extend(
            Event(47.0, 1, "rental", "paired demand pulse") for _ in range(8)
        )
        events.extend(
            Event(57.0, 2, "return", "paired demand pulse") for _ in range(8)
        )
    kind_order = {"return": 0, "rental": 1}
    events.sort(
        key=lambda event: (
            event.time_minutes,
            event.station_index,
            kind_order[event.kind],
        )
    )
    trace = EventTrace(
        events=tuple(events),
        source_label=(
            f"Poisson completed-event model, seed {seed}"
            + (", paired demand pulse" if paired_pulse else "")
        ),
        seed=seed,
        includes_paired_pulse=paired_pulse,
    )
    trace.validate(scenario)
    return trace


def _apply_action(
    scenario: BixiScenario,
    inventory: np.ndarray,
    truck_inventory: int,
    truck_station: int,
    action: BixiAction,
) -> tuple[np.ndarray, int, int, int]:
    if not 0 <= action.destination < len(scenario.stations):
        raise ValueError("controller requested an invalid destination")
    requested = int(action.transfer)
    if requested >= 0:
        realized = int(
            min(
                requested,
                scenario.transfer_limit,
                truck_inventory,
                scenario.capacities[truck_station] - inventory[truck_station],
            )
        )
    else:
        realized = -int(
            min(
                -requested,
                scenario.transfer_limit,
                scenario.truck_capacity - truck_inventory,
                inventory[truck_station],
            )
        )
    updated = inventory.copy()
    updated[truck_station] += realized
    return updated, truck_inventory - realized, action.destination, realized


def make_open_loop_controller(
    scenario: BixiScenario,
    profile: CompletedEventProfile,
) -> FrozenScheduleController:
    """Freeze the feedback heuristic on the deterministic mean-flow model."""

    feedback = InventoryFeedbackController(scenario, profile)
    inventory = scenario.initial_inventory.astype(float).copy()
    truck_inventory = scenario.initial_truck_inventory
    truck_station = scenario.initial_truck_station
    actions: list[BixiAction] = []
    for decision in range(scenario.decision_count):
        observation = BixiObservation(
            decision_index=decision,
            time_minutes=decision * scenario.control_period_minutes,
            station_inventory=inventory.copy(),
            truck_inventory=truck_inventory,
            truck_station=truck_station,
        )
        action = feedback(observation)
        actions.append(action)
        inventory, truck_inventory, truck_station, _ = _apply_action(
            scenario,
            inventory,
            truck_inventory,
            truck_station,
            action,
        )
        accepted = np.minimum(
            profile.completed_ends[decision],
            scenario.capacities - inventory,
        )
        inventory += accepted
        served = np.minimum(profile.completed_starts[decision], inventory)
        inventory -= served
    return FrozenScheduleController(actions)


def make_controller(
    name: ControllerName,
    scenario: BixiScenario,
    profile: CompletedEventProfile,
) -> Controller:
    if name == "none":
        return NoRelocationController()
    if name == "open_loop":
        return make_open_loop_controller(scenario, profile)
    if name == "feedback":
        return InventoryFeedbackController(scenario, profile)
    raise ValueError(f"unknown controller {name!r}")


def simulate(
    scenario: BixiScenario,
    trace: EventTrace,
    controller: Controller,
) -> BixiTrajectory:
    scenario.validate()
    trace.validate(scenario)

    inventory = scenario.initial_inventory.astype(float).copy()
    truck_inventory = scenario.initial_truck_inventory
    truck_station = scenario.initial_truck_station
    period = scenario.control_period_minutes
    decisions = scenario.decision_count

    checkpoints = [inventory.copy()]
    truck_checkpoints = [truck_inventory]
    location_checkpoints = [truck_station]
    requested_transfers: list[int] = []
    realized_transfers: list[int] = []
    destinations: list[int] = []
    cumulative_lost = [0]
    cumulative_rejected = [0]
    cumulative_moved = [0]
    cumulative_distance = [0.0]

    event_times: list[float] = []
    event_stations: list[int] = []
    event_kinds: list[EventKind] = []
    event_accepted: list[bool] = []
    event_inventory_after: list[float] = []

    history_times = [0.0]
    history_inventory = [inventory.copy()]
    history_truck = [truck_inventory]

    attempted_rentals = served_rentals = 0
    attempted_returns = accepted_returns = 0
    lost_rentals = rejected_returns = 0
    bikes_moved = unrealized_transfer_bikes = 0
    distance_km = 0.0
    event_index = 0

    for decision in range(decisions):
        start = float(decision * period)
        stop = float((decision + 1) * period)
        observation = BixiObservation(
            decision_index=decision,
            time_minutes=start,
            station_inventory=inventory.copy(),
            truck_inventory=truck_inventory,
            truck_station=truck_station,
        )
        action = controller(observation)
        previous_station = truck_station
        inventory, truck_inventory, truck_station, realized = _apply_action(
            scenario,
            inventory,
            truck_inventory,
            truck_station,
            action,
        )
        requested_transfers.append(int(action.transfer))
        realized_transfers.append(realized)
        destinations.append(truck_station)
        unrealized_transfer_bikes += abs(int(action.transfer) - realized)
        bikes_moved += abs(realized)
        distance_km += float(scenario.distance_km[previous_station, truck_station])
        history_times.append(start)
        history_inventory.append(inventory.copy())
        history_truck.append(truck_inventory)

        while event_index < len(trace.events):
            event = trace.events[event_index]
            if event.time_minutes >= stop:
                break
            station = event.station_index
            accepted = False
            if event.kind == "rental":
                attempted_rentals += 1
                if inventory[station] >= 1.0:
                    inventory[station] -= 1.0
                    served_rentals += 1
                    accepted = True
                else:
                    lost_rentals += 1
            else:
                attempted_returns += 1
                if inventory[station] < scenario.capacities[station]:
                    inventory[station] += 1.0
                    accepted_returns += 1
                    accepted = True
                else:
                    rejected_returns += 1
            event_times.append(event.time_minutes)
            event_stations.append(station)
            event_kinds.append(event.kind)
            event_accepted.append(accepted)
            event_inventory_after.append(float(inventory[station]))
            history_times.append(event.time_minutes)
            history_inventory.append(inventory.copy())
            history_truck.append(truck_inventory)
            event_index += 1

        checkpoints.append(inventory.copy())
        truck_checkpoints.append(truck_inventory)
        location_checkpoints.append(truck_station)
        cumulative_lost.append(lost_rentals)
        cumulative_rejected.append(rejected_returns)
        cumulative_moved.append(bikes_moved)
        cumulative_distance.append(distance_km)

    if event_index != len(trace.events):
        raise RuntimeError("simulation ended before all events were processed")
    if np.any(inventory < 0.0) or np.any(inventory > scenario.capacities):
        raise RuntimeError("simulation violated station bounds")
    if not 0 <= truck_inventory <= scenario.truck_capacity:
        raise RuntimeError("simulation violated the truck bound")

    initial_total = float(np.sum(scenario.initial_inventory)) + scenario.initial_truck_inventory
    expected_total = initial_total + accepted_returns - served_rentals
    actual_total = float(np.sum(inventory)) + truck_inventory
    if not np.isclose(expected_total, actual_total):
        raise RuntimeError("inventory conservation check failed")

    terminal_imbalance = float(
        np.sum(np.abs(inventory - 0.5 * scenario.capacities)) + truck_inventory
    )
    service_failures = lost_rentals + rejected_returns
    objective = float(
        scenario.failure_weight * service_failures
        + scenario.bike_moved_weight * bikes_moved
        + scenario.distance_weight * distance_km
        + scenario.terminal_imbalance_weight * terminal_imbalance
    )
    metrics = BixiMetrics(
        attempted_rentals=attempted_rentals,
        served_rentals=served_rentals,
        attempted_returns=attempted_returns,
        accepted_returns=accepted_returns,
        lost_rentals=lost_rentals,
        rejected_returns=rejected_returns,
        service_failures=service_failures,
        bikes_moved=bikes_moved,
        truck_distance_km=distance_km,
        unrealized_transfer_bikes=unrealized_transfer_bikes,
        terminal_imbalance=terminal_imbalance,
        terminal_truck_inventory=truck_inventory,
        objective=objective,
    )
    return BixiTrajectory(
        controller_name=controller.name,
        source_label=trace.source_label,
        checkpoint_time_minutes=np.arange(decisions + 1, dtype=float) * period,
        station_inventory=np.asarray(checkpoints, dtype=float),
        truck_inventory=np.asarray(truck_checkpoints, dtype=int),
        truck_station=np.asarray(location_checkpoints, dtype=int),
        requested_transfer=np.asarray(requested_transfers, dtype=int),
        realized_transfer=np.asarray(realized_transfers, dtype=int),
        destination=np.asarray(destinations, dtype=int),
        cumulative_lost_rentals=np.asarray(cumulative_lost, dtype=int),
        cumulative_rejected_returns=np.asarray(cumulative_rejected, dtype=int),
        cumulative_bikes_moved=np.asarray(cumulative_moved, dtype=int),
        cumulative_distance_km=np.asarray(cumulative_distance, dtype=float),
        event_time_minutes=np.asarray(event_times, dtype=float),
        event_station=np.asarray(event_stations, dtype=int),
        event_kind=tuple(event_kinds),
        event_accepted=np.asarray(event_accepted, dtype=bool),
        event_inventory_after=np.asarray(event_inventory_after, dtype=float),
        history_time_minutes=np.asarray(history_times, dtype=float),
        history_station_inventory=np.asarray(history_inventory, dtype=float),
        history_truck_inventory=np.asarray(history_truck, dtype=int),
        metrics=metrics,
    )


def simulate_fluid(
    scenario: BixiScenario,
    profile: CompletedEventProfile,
    controller: Controller,
) -> FluidTrajectory:
    """Run the deterministic fractional-flow approximation."""

    inventory = scenario.initial_inventory.astype(float).copy()
    truck_inventory = scenario.initial_truck_inventory
    truck_station = scenario.initial_truck_station
    checkpoints = [inventory.copy()]
    cumulative_lost = [0.0]
    cumulative_rejected = [0.0]
    lost = rejected = 0.0
    for decision in range(scenario.decision_count):
        observation = BixiObservation(
            decision_index=decision,
            time_minutes=decision * scenario.control_period_minutes,
            station_inventory=inventory.copy(),
            truck_inventory=truck_inventory,
            truck_station=truck_station,
        )
        action = controller(observation)
        inventory, truck_inventory, truck_station, _ = _apply_action(
            scenario,
            inventory,
            truck_inventory,
            truck_station,
            action,
        )
        attempted_returns = profile.completed_ends[decision]
        accepted_returns = np.minimum(
            attempted_returns,
            scenario.capacities - inventory,
        )
        rejected += float(np.sum(attempted_returns - accepted_returns))
        inventory += accepted_returns
        attempted_rentals = profile.completed_starts[decision]
        served_rentals = np.minimum(attempted_rentals, inventory)
        lost += float(np.sum(attempted_rentals - served_rentals))
        inventory -= served_rentals
        checkpoints.append(inventory.copy())
        cumulative_lost.append(lost)
        cumulative_rejected.append(rejected)
    return FluidTrajectory(
        checkpoint_time_minutes=np.arange(scenario.decision_count + 1, dtype=float)
        * scenario.control_period_minutes,
        station_inventory=np.asarray(checkpoints),
        cumulative_lost_rentals=np.asarray(cumulative_lost),
        cumulative_rejected_returns=np.asarray(cumulative_rejected),
    )


def make_censoring_counterexample() -> dict[str, object]:
    """Two latent request sequences with one identical completed-rental log."""

    time = np.arange(8, dtype=int)
    demand_stops = np.asarray([2, 2, 0, 0, 0, 0, 0, 0], dtype=int)
    demand_continues = np.asarray([2, 2, 2, 2, 2, 2, 2, 2], dtype=int)

    def observe(demand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inventory = 4
        completed: list[int] = []
        inventories = [inventory]
        for requested in demand:
            served = min(int(requested), inventory)
            completed.append(served)
            inventory -= served
            inventories.append(inventory)
        return np.asarray(completed), np.asarray(inventories)

    completed_a, inventory_a = observe(demand_stops)
    completed_b, inventory_b = observe(demand_continues)
    if not np.array_equal(completed_a, completed_b):
        raise RuntimeError("the counterexample no longer has identical logs")
    return {
        "time": time.tolist(),
        "demand_stops": demand_stops.tolist(),
        "demand_continues": demand_continues.tolist(),
        "completed": completed_a.tolist(),
        "inventory": inventory_a.tolist(),
        "logs_identical": bool(
            np.array_equal(completed_a, completed_b)
            and np.array_equal(inventory_a, inventory_b)
        ),
    }


def sha256(path: Path | str) -> str:
    """Public checksum helper used by artifact builders and tests."""

    return _sha256(Path(path))


__all__ = [
    "BixiAction",
    "BixiMetrics",
    "BixiObservation",
    "BixiScenario",
    "BixiTrajectory",
    "CompletedEventProfile",
    "Controller",
    "Event",
    "EventTrace",
    "FluidTrajectory",
    "FrozenScheduleController",
    "InventoryFeedbackController",
    "NoRelocationController",
    "Station",
    "load_completed_profile",
    "load_event_trace",
    "load_scenario",
    "make_censoring_counterexample",
    "make_controller",
    "make_open_loop_controller",
    "sample_poisson_trace",
    "sha256",
    "simulate",
    "simulate_fluid",
]
