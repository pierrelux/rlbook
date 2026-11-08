"""Wait or burn: collision avoidance learned from logged conjunction data messages.

When a satellite and another object are predicted to pass close to each
other, the operator receives a conjunction data message (CDM) roughly every
eight hours over the week before the time of closest approach (TCA). Each
message carries a collision probability estimated from the latest tracking.
The data are the CDMs that ESA's Space Debris Office released for the
Collision Avoidance Challenge (Uriot et al. 2022, Astrodynamics 6: 121-140;
dataset doi:10.5281/zenodo.4463683, CC BY 4.0).
data/conjunctions/kelvins_cdm_extract.csv.gz keeps a few columns of the
training file for all 13,154 events.

The decision problem is a teaching model built on these sequences. Decisions
are taken at every message received at least DEADLINE days before TCA, since
ESA plans avoidance maneuvers at least two days ahead (Uriot et al.). A
maneuver at t days before TCA costs (1 + 2 / t) / 2, in units of a maneuver
two days out: closer to the encounter, a larger velocity change is needed for
the same miss distance, and there is less time to plan. A satellite that
never maneuvers pays COLLISION_COST times the collision probability of the
last message of the event, the best estimate available. COLLISION_COST is
set so that a maneuver two days out breaks even at a probability of 1e-5, the
lower end of the reaction thresholds (1e-5 to 1e-4) of ESA missions in low
orbit (Uriot et al.).

Waiting does not change which messages arrive, and a maneuver ends the
problem at a known cost. Every logged sequence is therefore a sample of what
follows "wait", and fitted Q-iteration can learn the value of waiting from the
log alone, without a model of how collision estimates evolve.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "conjunctions" / "kelvins_cdm_extract.csv.gz"
DEADLINE = 2.0            # days before TCA after which no maneuver can be planned
COLLISION_COST = 1e5      # in units of a maneuver two days before TCA
RISK_FLOOR = -12.0        # log10 probabilities below this are treated as this (the data floor is -30)


def burn_cost(t):
    """Cost of a maneuver t days before closest approach."""
    return 0.5 * (1.0 + 2.0 / np.asarray(t, dtype=float))


@dataclass
class Epochs:
    """Decision epochs: the messages received before the deadline, then the deadline itself."""
    frame: pd.DataFrame
    event: np.ndarray          # event id of each row
    t: np.ndarray              # days to closest approach
    risk: np.ndarray           # log10 collision probability, floored at RISK_FLOOR
    burn: np.ndarray           # cost of a maneuver now
    outcome: np.ndarray        # cost of never maneuvering: COLLISION_COST * final probability
    last: np.ndarray           # True at the deadline epoch, the last decision of an event
    next: np.ndarray           # index of the next decision epoch of the same event (-1 if last)

    def features(self, names) -> np.ndarray:
        return self.frame[list(names)].to_numpy(dtype=float)


def load_epochs(path: Path = DATA) -> Epochs:
    """Decision epochs of every event with at least one message before the deadline.

    Each message received at least DEADLINE days before TCA is an epoch. Every
    event also gets a final epoch at the deadline itself, where the operator
    decides with the latest message; its "age" is how long ago that message
    arrived. After the deadline the outcome is the collision cost of the last
    message of the event.
    """
    d = pd.read_csv(path).sort_values(["event_id", "time_to_tca"], ascending=[True, False])
    final_risk = d.groupby("event_id").risk.last()
    m = d[d.time_to_tca >= DEADLINE].copy()
    m["age"] = 0.0
    deadline = m.groupby("event_id").tail(1).copy()
    deadline["age"] = deadline.time_to_tca - DEADLINE
    deadline["time_to_tca"] = DEADLINE
    m["at_deadline"], deadline["at_deadline"] = False, True
    e = (pd.concat([m, deadline]).sort_values(["event_id", "at_deadline", "time_to_tca"],
                                              ascending=[True, True, False]).reset_index(drop=True))
    e["risk"] = e.risk.clip(lower=RISK_FLOOR)
    e["final_risk"] = e.event_id.map(final_risk)
    last = e.at_deadline.to_numpy()
    nxt = np.arange(len(e)) + 1
    nxt[last] = -1
    return Epochs(
        frame=e, event=e.event_id.to_numpy(), t=e.time_to_tca.to_numpy(), risk=e.risk.to_numpy(),
        burn=burn_cost(e.time_to_tca.to_numpy()),
        outcome=COLLISION_COST * 10.0 ** np.minimum(e.final_risk.to_numpy(), 0.0),
        last=last, next=nxt,
    )


def realized_costs(epochs: Epochs, burn_here: np.ndarray) -> pd.DataFrame:
    """Per event: the cost of the policy that maneuvers at the first epoch where burn_here is True."""
    df = pd.DataFrame({"event": epochs.event, "burn_here": burn_here, "burn": epochs.burn,
                       "outcome": epochs.outcome, "t": epochs.t})
    first = df[df.burn_here].groupby("event").first()
    events = df.groupby("event").outcome.first().to_frame("outcome")
    events["maneuver"] = events.index.isin(first.index)
    events["maneuver_cost"] = first.burn.reindex(events.index).fillna(0.0)
    events["maneuver_t"] = first.t.reindex(events.index)
    events["collision_cost"] = np.where(events.maneuver, 0.0, events.outcome)
    events["cost"] = events.maneuver_cost + events.collision_cost
    return events


def summarize(events: pd.DataFrame, threshold: float = 1.0) -> dict:
    """Totals over events; a 'missed' event ends above the break-even probability without a maneuver."""
    return {
        "total_cost": float(events.cost.sum()),
        "maneuvers": int(events.maneuver.sum()),
        "maneuver_cost": float(events.maneuver_cost.sum()),
        "collision_cost": float(events.collision_cost.sum()),
        "missed": int(((~events.maneuver) & (events.outcome >= threshold)).sum()),
        "needed": int((events.outcome >= threshold).sum()),
    }


def rule_never(epochs: Epochs) -> np.ndarray:
    return np.zeros(epochs.t.size, dtype=bool)


def rule_as_soon_as_above(epochs: Epochs) -> np.ndarray:
    """Maneuver at the first message whose own probability makes a maneuver now worth its cost."""
    return COLLISION_COST * 10.0 ** epochs.risk > epochs.burn


def rule_at_deadline(epochs: Epochs) -> np.ndarray:
    """Wait until the deadline, then apply the same test to the latest message."""
    return epochs.last & (COLLISION_COST * 10.0 ** epochs.risk > epochs.burn)


def rule_hindsight(epochs: Epochs) -> np.ndarray:
    """Knows the final probability: maneuver at the first message if and only if it pays."""
    first = np.r_[True, epochs.event[1:] != epochs.event[:-1]]
    return first & (epochs.outcome > epochs.burn)


def fitted_q_iteration(epochs: Epochs, train: np.ndarray, features, relative: bool, fit_floor: float,
                       iterations: int = 30, n_estimators: int = 60, min_samples_leaf: int = 20,
                       seed: int = 0, n_jobs: int = -1, return_model: bool = False):
    """Learn W(s) = q(s, wait), the expected cost of waiting, from logged epochs; q(s, burn) is known.

    Targets: y = outcome at the last epoch of an event, min(burn', W(s')) otherwise.
    relative=False fits y directly. relative=True fits y / b(s) with b(s) = COLLISION_COST * p(s),
    the cost that the current message alone predicts, and returns b(s) times the fit.
    Only epochs with risk >= fit_floor are fitted; below it W is set to zero (never maneuver).
    Returns W at every epoch (the fit is applied to the held-out epochs too), and with
    return_model=True also the function that maps feature rows to W.
    """
    from sklearn.ensemble import ExtraTreesRegressor
    X = epochs.features(features)
    scale = COLLISION_COST * 10.0 ** epochs.risk if relative else np.ones(epochs.t.size)
    active = epochs.risk >= fit_floor
    rows = train & active
    nxt = np.maximum(epochs.next, 0)
    W = np.zeros(epochs.t.size)
    for k in range(iterations):
        y = np.where(epochs.last, epochs.outcome, np.minimum(epochs.burn[nxt], W[nxt]))
        model = ExtraTreesRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                    random_state=seed + k, n_jobs=n_jobs)
        model.fit(X[rows], (y / scale)[rows])
        model.set_params(n_jobs=1)     # a threaded predict sums the trees in a random order
        new = np.where(active, scale * model.predict(X), 0.0)
        change = np.abs(new - W).max()
        W = new
        if change < 1e-9:
            break
    if not return_model:
        return W
    risk_column = list(features).index("risk")

    def q_wait(points, model=model):
        points = np.asarray(points, dtype=float)
        scale = COLLISION_COST * 10.0 ** points[:, risk_column] if relative else 1.0
        return np.where(points[:, risk_column] >= fit_floor, scale * model.predict(points), 0.0)
    return W, q_wait


def folds_by_event(epochs: Epochs, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    events = np.unique(epochs.event)
    rng.shuffle(events)
    fold = dict(zip(events, np.arange(events.size) % n_folds))
    return np.array([fold[e] for e in epochs.event])


def cross_validated_fqi(epochs: Epochs, n_folds: int = 5, split_seed: int = 0, **options) -> np.ndarray:
    """W at every epoch, each fitted without the events of its own fold."""
    fold = folds_by_event(epochs, n_folds, split_seed)
    W = np.zeros(epochs.t.size)
    for i in range(n_folds):
        fitted = fitted_q_iteration(epochs, fold != i, **options)
        W[fold == i] = fitted[fold == i]
    return W
