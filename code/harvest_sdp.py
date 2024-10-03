import numpy as np
from scipy.interpolate import interp1d

# Parameters
r_max = 0.3
K = 125
T = 30  # Number of time steps
N_max = 100  # Maximum population size to consider
h_max = 0.5  # Maximum harvest rate
h_step = 0.1  # Step size for harvest rate

# Create state and decision spaces
N_space = np.linspace(1, N_max, 100)  # Using more granular state space
h_space = np.arange(0, h_max + h_step, h_step)

# Stochastic parameters
h_outcomes = np.array([0.75, 1.0, 1.25])
h_probs = np.array([0.25, 0.5, 0.25])
r_outcomes = np.array([0.85, 1.05, 1.15]) * r_max
r_probs = np.array([0.25, 0.5, 0.25])

# Initialize value function and policy
V = np.zeros((T + 1, len(N_space)))
policy = np.zeros((T, len(N_space)))

# State return function (F)
def state_return(N, h):
    return N * h

# State dynamics function (stochastic)
def state_dynamics(N, h, r):
    return N + r * N * (1 - N / K) - h * N

# Function to create interpolation function for a given time step
def create_interpolator(V_t, N_space):
    return interp1d(N_space, V_t, kind='linear', bounds_error=False, fill_value=(V_t[0], V_t[-1]))

# Backward iteration with stochastic dynamics
for t in range(T - 1, -1, -1):
    interpolator = create_interpolator(V[t+1], N_space)
    
    for i, N in enumerate(N_space):
        max_value = float('-inf')
        best_h = 0

        for h in h_space:
            if h > 1:  # Ensure harvest rate doesn't exceed 100%
                continue

            expected_value = 0
            for h_factor, h_prob in zip(h_outcomes, h_probs):
                for r_factor, r_prob in zip(r_outcomes, r_probs):
                    realized_h = h * h_factor
                    realized_r = r_factor

                    next_N = state_dynamics(N, realized_h, realized_r)
                    if next_N < 1:  # Ensure population doesn't go extinct
                        continue

                    # Use interpolation to get the value for next_N
                    value = state_return(N, realized_h) + interpolator(next_N)
                    expected_value += value * h_prob * r_prob

            if expected_value > max_value:
                max_value = expected_value
                best_h = h

        V[t, i] = max_value
        policy[t, i] = best_h

# Function to simulate the optimal policy using interpolation (stochastic version)
def simulate_optimal_policy(initial_N, T, num_simulations=100):
    all_trajectories = []
    all_harvests = []

    for _ in range(num_simulations):
        trajectory = [initial_N]
        harvests = []

        for t in range(T):
            N = trajectory[-1]
            
            # Create interpolator for the policy at time t
            policy_interpolator = interp1d(N_space, policy[t], kind='linear', bounds_error=False, fill_value=(policy[t][0], policy[t][-1]))
            
            intended_h = policy_interpolator(N)
            
            # Apply stochasticity
            h_factor = np.random.choice(h_outcomes, p=h_probs)
            r_factor = np.random.choice(r_outcomes, p=r_probs)
            
            realized_h = intended_h * h_factor
            harvests.append(N * realized_h)

            next_N = state_dynamics(N, realized_h, r_factor)
            trajectory.append(next_N)

        all_trajectories.append(trajectory)
        all_harvests.append(harvests)

    return all_trajectories, all_harvests

# Example usage
initial_N = 50
trajectories, harvests = simulate_optimal_policy(initial_N, T)

# Calculate average trajectory and total harvest
avg_trajectory = np.mean(trajectories, axis=0)
avg_total_harvest = np.mean([sum(h) for h in harvests])

print("Optimal policy (first few rows):")
print(policy[:5])
print("\nAverage population trajectory:", avg_trajectory)
print("Average total harvest:", avg_total_harvest)

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(121)
for traj in trajectories[:20]:  # Plot first 20 trajectories
    plt.plot(range(T+1), traj, alpha=0.3)
plt.plot(range(T+1), avg_trajectory, 'r-', linewidth=2)
plt.title('Population Trajectories')
plt.xlabel('Time')
plt.ylabel('Population')

plt.subplot(122)
plt.hist([sum(h) for h in harvests], bins=20)
plt.title('Distribution of Total Harvest')
plt.xlabel('Total Harvest')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()