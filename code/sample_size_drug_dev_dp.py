import numpy as np
from scipy.stats import binom
from scipy.special import ndtr as norm_cdf

def binomial_pmf(k, n, p):
    return binom.pmf(k, n, p)

def transition_prob_phase1(n1, eta1, p0):
    return np.sum([binomial_pmf(i, n1, p0) for i in range(int(eta1 * n1) + 1)])

def transition_prob_phase2(n2, eta2, delta):
    return norm_cdf((np.sqrt(n2) / 2) * delta - norm_cdf(1 - eta2))

def transition_prob_phase3(n3, eta3, delta):
    return norm_cdf((np.sqrt(n3) / 2) * delta - norm_cdf(1 - eta3))

def immediate_reward(n):
    return -n  # Changed to negative to represent cost

def value_iteration(S, A, gamma, g4, p0, delta, eta1, eta2, eta3, max_iter=1000, epsilon=1e-6):
    V = np.zeros(len(S))
    V[3] = g4  # Value for NDA approval state
    optimal_n = [None] * 3  # Store optimal n for each phase

    for _ in range(max_iter):
        V_old = V.copy()

        for i in range(2, -1, -1):  # Iterate backwards from Phase III to Phase I
            max_value = -np.inf

            for n in A:
                if i == 0:  # Phase I
                    p = transition_prob_phase1(n, eta1, p0)
                elif i == 1:  # Phase II
                    p = transition_prob_phase2(n, eta2, delta)
                else:  # Phase III
                    p = transition_prob_phase3(n, eta3, delta)

                value = immediate_reward(n) + gamma * p * V[i+1]

                if value > max_value:
                    max_value = value
                    optimal_n[i] = n

            V[i] = max_value

        if np.max(np.abs(V - V_old)) < epsilon:
            break

    return V, optimal_n

# Set up the problem parameters
S = ['Phase I', 'Phase II', 'Phase III', 'NDA approval']
A = range(10, 1001)
gamma = 0.95
g4 = 10000
p0 = 0.1  # Example toxicity rate for Phase I
delta = 0.5  # Example normalized treatment difference
eta1, eta2, eta3 = 0.2, 0.1, 0.025

# Run the value iteration algorithm
V, optimal_n = value_iteration(S, A, gamma, g4, p0, delta, eta1, eta2, eta3)

# Print results
for i, state in enumerate(S):
    print(f"Value for {state}: {V[i]:.2f}")
print(f"Optimal sample sizes: Phase I: {optimal_n[0]}, Phase II: {optimal_n[1]}, Phase III: {optimal_n[2]}")

# Sanity checks
print("\nSanity checks:")
print(f"1. NDA approval value: {V[3]}")
print(f"2. All values non-negative and <= NDA value: {all(0 <= v <= V[3] for v in V)}")
print(f"3. Optimal sample sizes in range: {all(10 <= n <= 1000 for n in optimal_n if n is not None)}")
