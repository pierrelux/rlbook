import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# System parameters
gamma, B, H, psi_c0, W = 0.5, 1, 0.18, 0.3, 0.25

# Simulation parameters
T = 50  # Total simulation time
dt = 0.1  # Time step
t = np.arange(0, T + dt, dt)
N = len(t)

# Number of trajectories
num_trajectories = 10

def psi_e(x1):
    return psi_c0 + H * (1 + 1.5 * ((x1 / W) - 1) - 0.5 * ((x1 / W) - 1)**3)

def phi(x2):
    return gamma * np.sign(x2) * np.sqrt(np.abs(x2))

def system_dynamics(t, x, u):
    x1, x2 = x
    dx1dt = B * (psi_e(x1) - x2 - u)
    dx2dt = (1 / B) * (x1 - phi(x2))
    return [dx1dt, dx2dt]

# "Do nothing" controller with small random noise
def u_func(t):
    return np.random.normal(0, 0.01)  # Mean 0, standard deviation 0.01

# Function to simulate a single trajectory
def simulate_trajectory(x0):
    sol = solve_ivp(lambda t, x: system_dynamics(t, x, u_func(t)), [0, T], x0, t_eval=t, method='RK45')
    return sol.y[0], sol.y[1]

# Generate multiple trajectories
trajectories = []
initial_conditions = []

for i in range(num_trajectories):
    # Randomize initial conditions around [0.5, 0.5]
    x0 = np.array([0.5, 0.5]) + np.random.normal(0, 0.05, 2)
    initial_conditions.append(x0)
    x1, x2 = simulate_trajectory(x0)
    trajectories.append((x1, x2))

# Calculate control inputs (small random noise)
u = np.array([u_func(ti) for ti in t])

# Plotting
plt.figure(figsize=(15, 15))

# State variables over time
plt.subplot(3, 1, 1)
for i, (x1, x2) in enumerate(trajectories):
    plt.plot(t, x1, label=f'x1 (Traj {i+1})' if i == 0 else "_nolegend_")
    plt.plot(t, x2, label=f'x2 (Traj {i+1})' if i == 0 else "_nolegend_")
plt.xlabel('Time')
plt.ylabel('State variables')
plt.title('State variables over time (Multiple Trajectories)')
plt.legend()
plt.grid(True)

# Phase portrait
plt.subplot(3, 1, 2)
for x1, x2 in trajectories:
    plt.plot(x1, x2)
    plt.plot(x1[0], x2[0], 'bo', markersize=5)
    plt.plot(x1[-1], x2[-1], 'ro', markersize=5)
plt.xlabel('x1 (mass flow)')
plt.ylabel('x2 (pressure)')
plt.title('Phase portrait (Multiple Trajectories)')
plt.grid(True)

# Control input (small random noise)
plt.subplot(3, 1, 3)
plt.plot(t, u, 'k-')
plt.xlabel('Time')
plt.ylabel('Control input (u)')
plt.title('Control input over time (Small random noise)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Save the data
np.savez('_static/compressor_surge_data_multi.npz', t=t, trajectories=trajectories, u=u, initial_conditions=initial_conditions)

print("Data collection complete. Results saved to 'compressor_surge_data_multi.npz'")
print(f"Data shape: {num_trajectories} trajectories, each with {N} time steps")
print(f"Time range: 0 to {T} seconds")
print("Initial conditions:")
for i, x0 in enumerate(initial_conditions):
    print(f"  Trajectory {i+1}: x1 = {x0[0]:.4f}, x2 = {x0[1]:.4f}")