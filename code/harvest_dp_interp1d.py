import numpy as np
from scipy.interpolate import interp1d

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

# Function to create interpolation function for a given time step
def create_interpolator(V_t, N_space):
    return interp1d(N_space, V_t, kind='cubic', bounds_error=False, fill_value=(V_t[0], V_t[-1]))

# Backward iteration with interpolation
for t in range(T - 1, -1, -1):
    interpolator = create_interpolator(V[t+1], N_space)
    
    for i, N in enumerate(N_space):
        max_value = float('-inf')
        best_h = 0

        for h in h_space:
            if h > 1:  # Ensure harvest rate doesn't exceed 100%
                continue

            next_N = state_dynamics(N, h)
            if next_N < 1:  # Ensure population doesn't go extinct
                continue

            # Use interpolation to get the value for next_N
            value = state_return(N, h) + interpolator(next_N)

            if value > max_value:
                max_value = value
                best_h = h

        V[t, i] = max_value
        policy[t, i] = best_h

# Function to simulate the optimal policy using interpolation
def simulate_optimal_policy(initial_N, T):
    trajectory = [initial_N]
    harvests = []

    for t in range(T):
        N = trajectory[-1]
        
        # Create interpolator for the policy at time t
        policy_interpolator = interp1d(N_space, policy[t], kind='cubic', bounds_error=False, fill_value=(policy[t][0], policy[t][-1]))
        
        h = policy_interpolator(N)
        harvests.append(float(N * h))  # Ensure harvest is a Python float

        next_N = state_dynamics(N, h)
        trajectory.append(float(next_N))  # Ensure next population value is a Python float

    return trajectory, harvests

# Example usage
initial_N = 50
trajectory, harvests = simulate_optimal_policy(initial_N, T)

print("Optimal policy (first few rows):")
print(policy[:5])
print("\nPopulation trajectory:", trajectory)
print("Harvests:", harvests)
print("Total harvest:", sum(harvests))