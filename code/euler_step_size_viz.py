import numpy as np
import matplotlib.pyplot as plt

def f(y, t):
    """
    Derivative function for vertical motion under gravity.
    y[0] is position, y[1] is velocity.
    """
    g = 9.81  # acceleration due to gravity (m/s^2)
    return np.array([y[1], -g])

def euler_method(f, y0, t0, t_end, h):
    """
    Implement Euler's method for the entire time range.
    """
    t = np.arange(t0, t_end + h, h)
    y = np.zeros((len(t), 2))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + h * f(y[i-1], t[i-1])
    return t, y

def true_solution(t):
    """
    Analytical solution for the ballistic trajectory.
    """
    y0, v0 = 0, 20  # initial height and velocity
    g = 9.81
    return y0 + v0*t - 0.5*g*t**2, v0 - g*t

# Set up the problem
t0, t_end = 0, 4
y0 = np.array([0, 20])  # initial height = 0, initial velocity = 20 m/s

# Different step sizes
step_sizes = [1.0, 0.5, 0.1]
colors = ['r', 'g', 'b']
markers = ['o', 's', '^']

# True solution
t_fine = np.linspace(t0, t_end, 1000)
y_true, v_true = true_solution(t_fine)

# Plotting
plt.figure(figsize=(12, 8))

# Plot Euler approximations
for h, color, marker in zip(step_sizes, colors, markers):
    t, y = euler_method(f, y0, t0, t_end, h)
    plt.plot(t, y[:, 0], color=color, marker=marker, linestyle='--', 
             label=f'Euler h = {h}', markersize=6, markerfacecolor='none')

# Plot true solution last so it's on top
plt.plot(t_fine, y_true, 'k-', label='True trajectory', linewidth=2, zorder=10)

plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Height (m)', fontsize=12)
plt.title("Euler's Method: Effect of Step Size on Ballistic Trajectory Approximation", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)

# Add text to explain the effect of step size
plt.text(2.5, 15, "Smaller step sizes\nyield better approximations", 
         bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
         fontsize=10, ha='center', va='center')

plt.tight_layout()
plt.show()