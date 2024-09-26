import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# System parameters
gamma, B, H, psi_c0, W = 0.5, 1, 0.18, 0.3, 0.25
kappa = 0.08
T, N = 12, 20  # Number of collocation points
t = np.linspace(0, T, N)
dt = T / (N - 1)
x1_star, x2_star = 0.40, 0.60

def psi_e(x1):
    return psi_c0 + H * (1 + 1.5 * ((x1 / W) - 1) - 0.5 * ((x1 / W) - 1)**3)

def phi(x2):
    return gamma * np.sign(x2) * np.sqrt(np.abs(x2))

def system_dynamics(t, x, u_func):
    x1, x2 = x
    u = u_func(t)
    dx1dt = B * (psi_e(x1) - x2 - u)
    dx2dt = (1 / B) * (x1 - phi(x2))
    return [dx1dt, dx2dt]

def objective(z):
    x = z[:2*N].reshape((N, 2))
    u = z[2*N:]
    
    # Trapezoidal rule for the cost function
    cost = 0
    for i in range(N-1):
        cost += 0.5 * dt * (kappa * u[i]**2 + kappa * u[i+1]**2)
    
    return cost

def constraints(z):
    x = z[:2*N].reshape((N, 2))
    u = z[2*N:]
    
    cons = []
    
    # Dynamics constraints (trapezoidal rule)
    for i in range(N-1):
        f_i = system_dynamics(t[i], x[i], lambda t: u[i])
        f_ip1 = system_dynamics(t[i+1], x[i+1], lambda t: u[i+1])
        cons.extend(x[i+1] - x[i] - 0.5 * dt * (np.array(f_i) + np.array(f_ip1)))
    
    # Terminal constraint
    cons.extend([x[-1, 0] - x1_star, x[-1, 1] - x2_star])
    
    # Initial condition constraint
    cons.extend([x[0, 0] - x0[0], x[0, 1] - x0[1]])
    
    return np.array(cons)

def solve_trajectory_optimization(x0):
    # Initial guess
    x_init = np.linspace(x0, [x1_star, x2_star], N)
    u_init = np.zeros(N)
    z0 = np.concatenate([x_init.flatten(), u_init])
    
    # Bounds
    bounds = [(None, None)] * (2*N)  # State variables
    bounds += [(0, 0.3)] * N  # Control inputs
    
    # Constraints
    cons = {'type': 'eq', 'fun': constraints}
    
    result = minimize(
        objective,
        z0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'disp': True, 'maxiter': 1000, 'ftol': 1e-6}
    )
    return result.x, result

# Run optimization
x0 = np.array([0.5, 0.5])
z_opt, result = solve_trajectory_optimization(x0)
x_opt_coll = z_opt[:2*N].reshape((N, 2))
u_opt_coll = z_opt[2*N:]

print(f"Optimization successful: {result.success}")
print(f"Final objective value: {result.fun}")
print(f"Final state: x1 = {x_opt_coll[-1, 0]:.4f}, x2 = {x_opt_coll[-1, 1]:.4f}")
print(f"Target state: x1 = {x1_star:.4f}, x2 = {x2_star:.4f}")

# Create interpolated control function
u_func = interp1d(t, u_opt_coll, kind='linear', bounds_error=False, fill_value=(u_opt_coll[0], u_opt_coll[-1]))

# Solve IVP with the optimized control
sol = solve_ivp(lambda t, x: system_dynamics(t, x, u_func), [0, T], x0, dense_output=True)

# Generate solution points
t_dense = np.linspace(0, T, 200)
x_ivp = sol.sol(t_dense).T

# Plotting
plt.figure(figsize=(15, 20))

# State variables over time
plt.subplot(3, 1, 1)
plt.plot(t, x_opt_coll[:, 0], 'bo-', label='x1 (collocation)')
plt.plot(t, x_opt_coll[:, 1], 'ro-', label='x2 (collocation)')
plt.plot(t_dense, x_ivp[:, 0], 'b--', label='x1 (integrated)')
plt.plot(t_dense, x_ivp[:, 1], 'r--', label='x2 (integrated)')
plt.axhline(y=x1_star, color='b', linestyle=':', label='x1 setpoint')
plt.axhline(y=x2_star, color='r', linestyle=':', label='x2 setpoint')
plt.xlabel('Time')
plt.ylabel('State variables')
plt.title('State variables over time')
plt.legend()
plt.grid(True)

# Phase portrait
plt.subplot(3, 1, 2)
plt.plot(x_opt_coll[:, 0], x_opt_coll[:, 1], 'go-', label='Collocation')
plt.plot(x_ivp[:, 0], x_ivp[:, 1], 'm--', label='Integrated')
plt.plot(x1_star, x2_star, 'r*', markersize=10, label='Setpoint')
plt.xlabel('x1 (mass flow)')
plt.ylabel('x2 (pressure)')
plt.title('Phase portrait')
plt.legend()
plt.grid(True)

# Control inputs
plt.subplot(3, 1, 3)
plt.step(t, u_opt_coll, 'g-', where='post', label='Collocation')
plt.plot(t_dense, u_func(t_dense), 'm--', label='Interpolated')
plt.xlabel('Time')
plt.ylabel('Control input (u)')
plt.title('Control input over time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()