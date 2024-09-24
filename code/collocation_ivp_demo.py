import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

def ode_function(y, t):
    """Define the ODE: dy/dt = -y"""
    return -y

def solve_ode_collocation(ode_func, t_span, y0, order):
    t0, tf = t_span
    n_points = order + 1  # number of collocation points
    t_points = np.linspace(t0, tf, n_points)
    
    def collocation_residuals(coeffs):
        residuals = []
        # Initial condition residual
        y_init = sum(c * t_points[0]**i for i, c in enumerate(coeffs))
        residuals.append(y_init - y0)
        # Collocation point residuals
        for t in t_points[1:]:  # Skip the first point as it's used for initial condition
            y = sum(c * t**i for i, c in enumerate(coeffs))
            dy_dt = sum(c * i * t**(i-1) for i, c in enumerate(coeffs) if i > 0)
            residuals.append(dy_dt - ode_func(y, t))
        return residuals

    # Initial guess for coefficients
    initial_coeffs = [y0] + [0] * order

    # Solve the system of equations
    solution = root(collocation_residuals, initial_coeffs)
    
    if not solution.success:
        raise ValueError("Failed to converge to a solution.")

    coeffs = solution.x

    # Generate solution
    t_fine = np.linspace(t0, tf, 100)
    y_solution = sum(c * t_fine**i for i, c in enumerate(coeffs))

    return t_fine, y_solution, t_points, coeffs

# Example usage
t_span = (0, 2)
y0 = 1
orders = [1, 2, 3, 4, 5]  # Different polynomial orders to try

plt.figure(figsize=(12, 8))

for order in orders:
    t, y, t_collocation, coeffs = solve_ode_collocation(ode_function, t_span, y0, order)
    
    # Calculate y values at collocation points
    y_collocation = sum(c * t_collocation**i for i, c in enumerate(coeffs))
    
    # Plot the results
    plt.plot(t, y, label=f'Order {order}')
    plt.scatter(t_collocation, y_collocation, s=50, zorder=5)

# Plot the analytical solution
t_analytical = np.linspace(t_span[0], t_span[1], 100)
y_analytical = y0 * np.exp(-t_analytical)
plt.plot(t_analytical, y_analytical, 'k--', label='Analytical')

plt.xlabel('t')
plt.ylabel('y')
plt.title('ODE Solutions: dy/dt = -y, y(0) = 1')
plt.legend()
plt.grid(True)
plt.show()

# Print error for each order
print("Maximum absolute errors:")
for order in orders:
    t, y, _, _ = solve_ode_collocation(ode_function, t_span, y0, order)
    y_true = y0 * np.exp(-t)
    max_error = np.max(np.abs(y - y_true))
    print(f"Order {order}: {max_error:.6f}")