import numpy as np
import matplotlib.pyplot as plt

def f(y, t):
    """
    Derivative function for vertical motion under gravity.
    y[0] is position, y[1] is velocity.
    """
    g = 9.81  # acceleration due to gravity (m/s^2)
    return np.array([y[1], -g])

def true_solution(t):
    """
    Analytical solution for the ballistic trajectory.
    """
    y0, v0 = 0, 20  # initial height and velocity
    g = 9.81
    return y0 + v0*t - 0.5*g*t**2, v0 - g*t

def trapezoid_method_visual(f, y0, t0, t_end, h):
    """
    Implement the trapezoid method for the entire time range.
    Returns predictor and corrector steps for visualization.
    """
    t = np.arange(t0, t_end + h, h)
    y = np.zeros((len(t), 2))
    y_predictor = np.zeros((len(t), 2))
    y[0] = y_predictor[0] = y0
    for i in range(1, len(t)):
        # Predictor step (Euler forward)
        slope_start = f(y[i-1], t[i-1])
        y_predictor[i] = y[i-1] + h * slope_start
        
        # Corrector step
        slope_end = f(y_predictor[i], t[i])
        y[i] = y[i-1] + h * 0.5 * (slope_start + slope_end)
    
    return t, y, y_predictor

# Set up the problem
t0, t_end = 0, 2
y0 = np.array([0, 20])  # initial height = 0, initial velocity = 20 m/s
h = 0.5  # Step size

# Compute trapezoid method steps
t, y_corrector, y_predictor = trapezoid_method_visual(f, y0, t0, t_end, h)

# Plotting
plt.figure(figsize=(12, 8))

# Plot the true solution for comparison
t_fine = np.linspace(t0, t_end, 1000)
y_true, v_true = true_solution(t_fine)
plt.plot(t_fine, y_true, 'k-', label='True trajectory', linewidth=1.5)

# Plot the predictor and corrector steps
for i in range(len(t)-1):
    # Points for the predictor step
    p0 = [t[i], y_corrector[i, 0]]
    p1_predictor = [t[i+1], y_predictor[i+1, 0]]
    
    # Points for the corrector step
    p1_corrector = [t[i+1], y_corrector[i+1, 0]]
    
    # Plot predictor step
    plt.plot([p0[0], p1_predictor[0]], [p0[1], p1_predictor[1]], 'r--', linewidth=2)
    plt.plot(p1_predictor[0], p1_predictor[1], 'ro', markersize=8)
    
    # Plot corrector step
    plt.plot([p0[0], p1_corrector[0]], [p0[1], p1_corrector[1]], 'g--', linewidth=2)
    plt.plot(p1_corrector[0], p1_corrector[1], 'go', markersize=8)
    
    # Add arrows to show the predictor and corrector adjustments
    plt.arrow(p0[0], p0[1], h, y_predictor[i+1, 0] - p0[1], color='r', width=0.005, 
              head_width=0.02, head_length=0.02, length_includes_head=True, zorder=5)
    plt.arrow(p1_predictor[0], p1_predictor[1], 0, y_corrector[i+1, 0] - y_predictor[i+1, 0], 
              color='g', width=0.005, head_width=0.02, head_length=0.02, length_includes_head=True, zorder=5)

# Add legend entries for predictor and corrector steps
plt.plot([], [], 'r--', label='Predictor step (Forward Euler)')
plt.plot([], [], 'g-', label='Corrector step (Trapezoid)')

# Labels and title
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Height (m)', fontsize=12)
plt.title("Trapezoid Method: Predictor-Corrector Structure", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()