import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load the data
data = np.load('_static/compressor_surge_data_multi.npz', allow_pickle=True)
t = data['t']
trajectories = data['trajectories']
u = data['u']
initial_conditions = data['initial_conditions']

# Known system parameters
gamma, H, psi_c0, W = 0.5, 0.18, 0.3, 0.25
# B is the parameter we want to identify
B_true = 1.0  # True value, used for comparison

def psi_e(x1):
    return psi_c0 + H * (1 + 1.5 * ((x1 / W) - 1) - 0.5 * ((x1 / W) - 1)**3)

def phi(x2):
    return gamma * np.sign(x2) * np.sqrt(np.abs(x2))

def system_dynamics(t, x, u, B):
    x1, x2 = x
    dx1dt = B * (psi_e(x1) - x2 - u)
    dx2dt = (1 / B) * (x1 - phi(x2))
    return np.array([dx1dt, dx2dt])

def rk4_step(f, t, x, u, dt, B):
    k1 = f(t, x, u, B)
    k2 = f(t + 0.5*dt, x + 0.5*dt*k1, u, B)
    k3 = f(t + 0.5*dt, x + 0.5*dt*k2, u, B)
    k4 = f(t + dt, x + dt*k3, u, B)
    return x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_trajectory(x0, B):
    x = np.zeros((len(t), 2))
    x[0] = x0
    for i in range(1, len(t)):
        x[i] = rk4_step(system_dynamics, t[i-1], x[i-1], u[i-1], t[i] - t[i-1], B)
    return x

def objective(B):
    error = 0
    for i, (x1_obs, x2_obs) in enumerate(trajectories):
        x_sim = simulate_trajectory(initial_conditions[i], B[0])
        error += np.sum((x_sim[:, 0] - x1_obs)**2 + (x_sim[:, 1] - x2_obs)**2)
    return error

# Perform optimization
result = minimize(objective, x0=[1.5], method='Nelder-Mead', options={'disp': True})

B_identified = result.x[0]

print(f"True B: {B_true}")
print(f"Identified B: {B_identified}")
print(f"Relative error: {abs(B_identified - B_true) / B_true * 100:.2f}%")

# Plot results
plt.figure(figsize=(15, 10))

# Plot one trajectory for comparison
traj_index = 0
x1_obs, x2_obs = trajectories[traj_index]
x_sim = simulate_trajectory(initial_conditions[traj_index], B_identified)

plt.subplot(2, 1, 1)
plt.plot(t, x1_obs, 'b-', label='Observed x1')
plt.plot(t, x2_obs, 'r-', label='Observed x2')
plt.plot(t, x_sim[:, 0], 'b--', label='Simulated x1')
plt.plot(t, x_sim[:, 1], 'r--', label='Simulated x2')
plt.xlabel('Time')
plt.ylabel('State variables')
plt.title('Observed vs Simulated Trajectory')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(x1_obs, x2_obs, 'g-', label='Observed')
plt.plot(x_sim[:, 0], x_sim[:, 1], 'm--', label='Simulated')
plt.xlabel('x1 (mass flow)')
plt.ylabel('x2 (pressure)')
plt.title('Phase Portrait: Observed vs Simulated')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()