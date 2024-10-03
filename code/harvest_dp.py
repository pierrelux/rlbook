import numpy as np

# Parameters
r_max = 0.3
K = 125
T = 20  # Number of time steps
N_max = 100  # Maximum population size to consider
h_max = 0.5  # Maximum harvest rate
h_step = 0.1  # Step size for harvest rate

# Create state and decision spaces
N_space = np.arange(1, N_max + 1)
h_space = np.arange(0, h_max + h_step, h_step)

# Initialize value function and policy
V = np.zeros((T + 1, len(N_space)))
policy = np.zeros((T, len(N_space)))

# Terminal value function (F_T)
def terminal_value(N):
    return 0

# State return function (F)
def state_return(N, h):
    return N * h

# State dynamics function
def state_dynamics(N, h):
    return N + r_max * N * (1 - N / K) - N * h

# Backward iteration
for t in range(T - 1, -1, -1):
    for i, N in enumerate(N_space):
        max_value = float('-inf')
        best_h = 0

        for h in h_space:
            if h > 1:  # Ensure harvest rate doesn't exceed 100%
                continue

            next_N = state_dynamics(N, h)
            if next_N < 1:  # Ensure population doesn't go extinct
                continue

            next_N_index = np.searchsorted(N_space, next_N)
            if next_N_index == len(N_space):
                next_N_index -= 1

            value = state_return(N, h) + V[t + 1, next_N_index]

            if value > max_value:
                max_value = value
                best_h = h

        V[t, i] = max_value
        policy[t, i] = best_h

# Function to simulate the optimal policy with conversion to Python floats
def simulate_optimal_policy(initial_N, T):
    trajectory = [float(initial_N)]  # Ensure first value is a Python float
    harvests = []

    for t in range(T):
        N = trajectory[-1]
        N_index = np.searchsorted(N_space, N)
        if N_index == len(N_space):
            N_index -= 1

        h = policy[t, N_index]
        harvests.append(float(N * h))  # Ensure harvest is a Python float

        next_N = state_dynamics(N, h)
        trajectory.append(float(next_N))  # Ensure next population value is a Python float

    return trajectory, harvests

# Example usage
initial_N = 50
trajectory, harvests = simulate_optimal_policy(initial_N, T)

print("Optimal policy:")
print(policy)
print("\nPopulation trajectory:", trajectory)
print("Harvests:", harvests)
print("Total harvest:", sum(harvests))