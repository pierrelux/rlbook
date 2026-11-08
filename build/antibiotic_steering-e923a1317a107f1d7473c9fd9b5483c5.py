"""Steering bacterial evolution with antibiotic sequences as a finite-horizon MDP.

The fitness landscapes are the growth rates measured by Mira et al. (2015) for
the 16 combinations of four TEM beta-lactamase mutations in E. coli, one
landscape per beta-lactam drug. Evolution under a drug follows the strong
selection, weak mutation (SSWM) Markov chain of Nichol et al. (2015): from the
current genotype the population moves to a uniformly chosen fitter one-mutation
neighbour (their r = 0 case) and stays put at a local fitness peak. A drug is
applied long enough for the chain to reach a peak, so its transition matrix is
the limit of the chain. Adapted from https://github.com/Daniel-Nichol/EvolutionarySteering
(MIT licence).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np

N_SITES = 4
N_GENOTYPES = 2 ** N_SITES
# Bit k of a genotype index (most significant first) marks the k-th mutation.
MUTATIONS = ("M69L", "E104K", "G238S", "N276D")

# Growth rates under each drug, indexed by genotype 0000, 0001, ..., 1111.
LANDSCAPES = {
    "AMP": ("Ampicillin", [1.851, 2.082, 1.948, 2.434, 2.024, 2.198, 2.033, 0.034, 1.57, 2.165, 0.051, 0.083, 2.186, 2.322, 0.088, 2.821]),
    "AM": ("Amoxicillin", [1.778, 1.782, 2.042, 1.752, 1.448, 1.544, 1.184, 0.063, 1.72, 2.008, 1.799, 2.005, 1.557, 2.247, 1.768, 2.047]),
    "CEC": ("Cefaclor", [2.258, 1.996, 2.151, 2.648, 2.396, 1.846, 2.23, 0.214, 0.234, 0.172, 2.242, 0.093, 2.15, 0.095, 2.64, 0.516]),
    "CTX": ("Cefotaxime", [0.16, 0.085, 1.936, 2.348, 1.653, 0.138, 2.295, 2.269, 0.185, 0.14, 1.969, 0.203, 0.225, 0.092, 0.119, 2.412]),
    "ZOX": ("Ceftizoxime", [0.993, 0.805, 2.069, 2.683, 1.698, 2.01, 2.138, 2.688, 1.106, 1.171, 1.894, 0.681, 1.116, 1.105, 1.103, 2.591]),
    "CXM": ("Cefuroxime", [1.748, 1.7, 2.07, 1.938, 2.94, 2.173, 2.918, 3.272, 0.423, 1.578, 1.911, 2.754, 2.024, 1.678, 1.591, 2.923]),
    "CRO": ("Ceftriaxone", [1.092, 0.287, 2.554, 3.042, 2.88, 0.656, 2.732, 0.436, 0.83, 0.54, 3.173, 1.153, 1.407, 0.751, 2.74, 3.227]),
    "AMC": ("Amoxicillin + clavulanate", [1.435, 1.573, 1.061, 1.457, 1.672, 1.625, 0.073, 0.068, 1.417, 1.351, 1.538, 1.59, 1.377, 1.914, 1.307, 1.728]),
    "CAZ": ("Ceftazidime", [2.134, 2.656, 2.618, 2.688, 2.042, 2.756, 2.924, 0.251, 0.288, 0.576, 1.604, 1.378, 2.63, 2.677, 2.893, 2.563]),
    "CTT": ("Cefotetan", [2.125, 1.922, 2.804, 0.588, 3.291, 2.888, 3.082, 3.508, 3.238, 2.966, 2.883, 0.89, 0.546, 3.181, 3.193, 2.543]),
    "SAM": ("Ampicillin + sulbactam", [1.879, 2.533, 0.133, 0.094, 2.456, 2.437, 0.083, 0.094, 2.198, 2.57, 2.308, 2.886, 2.504, 3.002, 2.528, 3.453]),
    "CPR": ("Cefprozil", [1.743, 1.662, 1.763, 1.785, 2.018, 2.05, 2.042, 0.218, 1.553, 0.256, 0.165, 0.221, 0.223, 0.239, 1.811, 0.288]),
    "CPD": ("Cefpodoxime", [0.595, 0.245, 2.604, 3.043, 1.761, 1.471, 2.91, 3.096, 0.432, 0.388, 2.651, 1.103, 0.638, 0.986, 0.963, 3.268]),
    "TZP": ("Piperacillin + tazobactam", [2.679, 2.906, 2.427, 0.141, 3.038, 3.309, 2.528, 0.143, 2.709, 2.5, 0.172, 0.093, 2.453, 2.739, 0.609, 0.171]),
    "FEP": ("Cefepime", [2.59, 2.572, 2.393, 2.832, 2.44, 2.808, 2.652, 0.611, 2.067, 2.446, 2.957, 2.633, 2.735, 2.863, 2.796, 3.203]),
}
DRUGS = tuple(LANDSCAPES)


def label(g: int) -> str:
    return format(g, f"0{N_SITES}b")


def neighbours(g: int) -> list[int]:
    return [g ^ (1 << (N_SITES - 1 - k)) for k in range(N_SITES)]


def fitness(drug: str) -> np.ndarray:
    return np.asarray(LANDSCAPES[drug][1])


def step_matrix(drug: str) -> np.ndarray:
    """One SSWM substitution: a uniformly chosen fitter neighbour, or stay at a peak."""
    f = fitness(drug)
    P = np.zeros((N_GENOTYPES, N_GENOTYPES))
    for g in range(N_GENOTYPES):
        fitter = [h for h in neighbours(g) if f[h] > f[g]]
        if fitter:
            P[g, fitter] = 1.0 / len(fitter)
        else:
            P[g, g] = 1.0
    return P


def limit_matrix(P: np.ndarray) -> np.ndarray:
    # Every walk climbs strictly in fitness, so it is absorbed after at most 15 steps.
    return np.linalg.matrix_power(P, N_GENOTYPES)


@lru_cache(maxsize=None)
def _transition(drug: str) -> np.ndarray:
    T = limit_matrix(step_matrix(drug))
    T.setflags(write=False)
    return T


def transition(drug: str) -> np.ndarray:
    """Genotype distribution after the drug has carried every walk to a peak."""
    return _transition(drug)


def resistant_genotype(drug: str) -> int:
    """The most resistant genotype: the global fitness peak of the final drug."""
    return int(np.argmax(fitness(drug)))


@dataclass(frozen=True)
class SteeringSolution:
    final_drug: str
    values: np.ndarray        # values[t, s]: probability of ending resistant from s before steering step t
    policy: np.ndarray        # policy[t, s]: index into DRUGS of the steering drug at step t


def backward_induction(final_drug: str, steps: int) -> SteeringSolution:
    """Minimize P(final genotype is the resistant peak) over `steps` steering drugs.

    The genotype is observed before each drug, so the policy is Markov:
    one drug per (step, genotype). The terminal value is the probability that
    the final treatment carries a population from s to the resistant peak.
    """
    T = np.stack([transition(d) for d in DRUGS])                  # (drug, s, s')
    v = transition(final_drug)[:, resistant_genotype(final_drug)]
    values, policy = [v], []
    for _ in range(steps):
        q = T @ v                                                   # q[a, s] = sum_j p(j|s,a) v(j)
        a = np.argmin(q, axis=0)
        v = q[a, np.arange(N_GENOTYPES)]
        values.append(v)
        policy.append(a)
    return SteeringSolution(final_drug, np.array(values[::-1]), np.array(policy[::-1]))


def open_loop_probability(sequence: tuple[str, ...], final_drug: str, start: np.ndarray) -> float:
    dist = start.copy()
    for d in sequence:
        dist = dist @ transition(d)
    dist = dist @ transition(final_drug)
    return float(dist[resistant_genotype(final_drug)])


def best_open_loop(final_drug: str, steps: int, start: np.ndarray) -> tuple[tuple[str, ...], float]:
    """Exhaustive search over the 15**steps fixed drug sequences."""
    best = ((), open_loop_probability((), final_drug, start))
    if steps == 0:
        return best
    for seq in product(DRUGS, repeat=steps):
        p = open_loop_probability(seq, final_drug, start)
        if p < best[1] - 1e-12:
            best = (seq, p)
    return best


def uniform_start() -> np.ndarray:
    return np.full(N_GENOTYPES, 1.0 / N_GENOTYPES)
