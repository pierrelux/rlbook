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

# Function to linearly interpolate between grid points in N_space
def interpolate_value_function(V, N_space, next_N, t):
    if next_N <= N_space[0]:
        return V[t, 0]  # Below or at minimum population, return minimum value
    if next_N >= N_space[-1]:
        return V[t, -1]  # Above or at maximum population, return maximum value
    
    # Find indices to interpolate between
    lower_idx = np.searchsorted(N_space, next_N) - 1
    upper_idx = lower_idx + 1
    
    # Linear interpolation
    N_lower = N_space[lower_idx]
    N_upper = N_space[upper_idx]
    weight = (next_N - N_lower) / (N_upper - N_lower)
    return (1 - weight) * V[t, lower_idx] + weight * V[t, upper_idx]

# Backward iteration with interpolation
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
            
            # Interpolate value for next_N
            value = state_return(N, h) + interpolate_value_function(V, N_space, next_N, t + 1)
            
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
        
        # Interpolate optimal harvest rate
        if N <= N_space[0]:
            h = policy[t, 0]
        elif N >= N_space[-1]:
            h = policy[t, -1]
        else:
            lower_idx = np.searchsorted(N_space, N) - 1
            upper_idx = lower_idx + 1
            weight = (N - N_space[lower_idx]) / (N_space[upper_idx] - N_space[lower_idx])
            h = (1 - weight) * policy[t, lower_idx] + weight * policy[t, upper_idx]
        
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