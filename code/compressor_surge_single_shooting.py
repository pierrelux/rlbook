import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# System parameters
gamma, B, H, psi_c0, W = 0.5, 1, 0.18, 0.3, 0.25
alpha, beta, kappa, R = 1, 0, 0.08, 0
T, N = 12, 60
dt = T / N
x1_star, x2_star = 0.40, 0.60

def psi_e(x1):
    return psi_c0 + H * (1 + 1.5 * ((x1 / W) - 1) - 0.5 * ((x1 / W) - 1)**3)

def phi(x2):
    return gamma * np.sign(x2) * np.sqrt(np.abs(x2))

def system_dynamics(x, u):
    x1, x2 = x
    dx1dt = B * (psi_e(x1) - x2 - u)
    dx2dt = (1 / B) * (x1 - phi(x2))
    return np.array([dx1dt, dx2dt])

def euler_step(x, u, dt):
    return x + dt * system_dynamics(x, u)

def instantenous_cost(x, u):
    return (alpha * np.sum((x - np.array([x1_star, x2_star]))**2) + kappa * u**2)

def terminal_cost(x):
    return beta * np.sum((x - np.array([x1_star, x2_star]))**2)

def objective_and_constraints(z):
    u, v = z[:-1], z[-1]
    x = np.zeros((N+1, 2))
    x[0] = x0
    obj = 0
    cons = []
    for i in range(N):
        x[i+1] = euler_step(x[i], u[i], dt)
        obj += dt * instantenous_cost(x[i], u[i])
        cons.append(0.4 - x[i+1, 1] - v)
    obj += terminal_cost(x[-1]) + R * v**2
    return obj, np.array(cons)

def solve_trajectory_optimization(x0, u_init):
    z0 = np.zeros(N + 1)
    z0[:-1] = u_init
    bounds = [(0, 0.3)] * N + [(0, None)]
    result = minimize(
        lambda z: objective_and_constraints(z)[0],
        z0,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'ineq', 'fun': lambda z: -objective_and_constraints(z)[1]},
        options={'disp': True, 'maxiter': 1000, 'ftol': 1e-6}
    )
    return result.x, result

def simulate_trajectory(x0, u):
    x = np.zeros((N+1, 2))
    x[0] = x0
    for i in range(N):
        x[i+1] = euler_step(x[i], u[i], dt)
    return x

# Run optimizations and simulations
x0 = np.array([0.25, 0.25])
t = np.linspace(0, T, N+1)

# Optimized control starting from zero
z_single_shooting, _ = solve_trajectory_optimization(x0, np.zeros(N))
u_opt_shoot, v_opt_shoot = z_single_shooting[:-1], z_single_shooting[-1]
x_opt_shoot = simulate_trajectory(x0, u_opt_shoot)

# Do-nothing control (u = 0)
u_nothing = np.zeros(N)
x_nothing = simulate_trajectory(x0, u_nothing)

# Plotting
plt.figure(figsize=(15, 20))

# State variables over time
plt.subplot(3, 1, 1)
plt.plot(t, x_opt_shoot[:, 0], label='x1 (opt from 0)')
plt.plot(t, x_opt_shoot[:, 1], label='x2 (opt from 0)')
plt.plot(t, x_nothing[:, 0], ':', label='x1 (do-nothing)')
plt.plot(t, x_nothing[:, 1], ':', label='x2 (do-nothing)')
plt.axhline(y=x1_star, color='r', linestyle='--', label='x1 setpoint')
plt.axhline(y=x2_star, color='g', linestyle='--', label='x2 setpoint')
plt.xlabel('Time')
plt.ylabel('State variables')
plt.title('State variables over time')
plt.legend()
plt.grid(True)

# Phase portrait
plt.subplot(3, 1, 2)
plt.plot(x_opt_shoot[:, 0], x_opt_shoot[:, 1], label='Optimized from 0')
plt.plot(x_nothing[:, 0], x_nothing[:, 1], ':', label='Do-nothing')
plt.plot(x1_star, x2_star, 'r*', markersize=10, label='Setpoint')
plt.xlabel('x1 (mass flow)')
plt.ylabel('x2 (pressure)')
plt.title('Phase portrait')
plt.legend()
plt.grid(True)

# Control inputs
plt.subplot(3, 1, 3)
plt.plot(t[:-1], u_opt_shoot, label='Optimized from 0')
plt.plot(t[:-1], u_nothing, ':', label='Do-nothing')
plt.xlabel('Time')
plt.ylabel('Control input (u)')
plt.title('Control input over time')
plt.legend()
plt.grid(True)
