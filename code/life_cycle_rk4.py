import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Model parameters
T = 50  # Time horizon
N = 1000  # Number of time steps
dt = T / N
rho = 0.05  # Discount rate
gamma = 2  # Risk aversion parameter

# Utility function
def u(c):
    return c**(1 - gamma) / (1 - gamma)

# Wage function
def w(t):
    return 1 + 0.1 * t - 0.001 * t**2

# Asset return function
def f(A):
    return 0.03 * A  # 3% return on assets

# Consumption function
def c(t, theta):
    return np.maximum(theta[0] + theta[1]*t + theta[2]*t**2 + theta[3]*t**3, 1e-6)

# Single step of RK4 method
def rk4_step(A, t, theta):
    k1 = f(A) + w(t) - c(t, theta)
    k2 = f(A + 0.5*dt*k1) + w(t + 0.5*dt) - c(t + 0.5*dt, theta)
    k3 = f(A + 0.5*dt*k2) + w(t + 0.5*dt) - c(t + 0.5*dt, theta)
    k4 = f(A + dt*k3) + w(t + dt) - c(t + dt, theta)
    return A + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

# Phi function: propagate dynamics from t=0 to t=T
def Phi(theta):
    A = 0  # Initial assets
    t = 0
    for _ in range(N):
        A = rk4_step(A, t, theta)
        t += dt
    return A

# Objective function
def objective(theta):
    t = np.linspace(0, T, N)
    consumption = c(t, theta)
    return -np.sum(np.exp(-rho * t) * u(consumption)) * dt

# Constraint function
def constraint(theta):
    return Phi(theta)

# Optimize consumption parameters
initial_guess = [1.0, 0.0, 0.0, 0.0]
cons = {'type': 'eq', 'fun': constraint}
result = minimize(objective, initial_guess, method='SLSQP', constraints=cons)
optimal_theta = result.x

# Simulate the optimal path
t = np.linspace(0, T, N)
A = np.zeros(N)
consumption = c(t, optimal_theta)

for i in range(1, N):
    A[i] = rk4_step(A[i-1], t[i-1], optimal_theta)

print(f"Optimal consumption parameters (theta): {optimal_theta}")
print(f"Final assets: {A[-1]:.4f}")

# Plotting
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(t, A)
plt.title('Assets over time')
plt.xlabel('Time')
plt.ylabel('Assets')

plt.subplot(2, 2, 2)
plt.plot(t, consumption)
plt.title('Consumption over time')
plt.xlabel('Time')
plt.ylabel('Consumption')

plt.subplot(2, 2, 3)
plt.plot(t, w(t))
plt.title('Wage over time')
plt.xlabel('Time')
plt.ylabel('Wage')

plt.subplot(2, 2, 4)
plt.plot(t, f(A))
plt.title('Asset returns over time')
plt.xlabel('Time')
plt.ylabel('Asset returns')

plt.tight_layout()
plt.show()