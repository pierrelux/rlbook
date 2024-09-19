import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

def v(t):
    """Velocity function for the ballistic trajectory."""
    v0 = 20   # initial velocity (m/s)
    g = 9.81  # acceleration due to gravity (m/s^2)
    return v0 - g * t

def position(t):
    """Position function (integral of velocity)."""
    v0 = 20
    g = 9.81
    return v0*t - 0.5*g*t**2

# Set up the problem
t0, t_end = 0, 2
num_points = 1000
t = np.linspace(t0, t_end, num_points)

# Calculate true velocity and position
v_true = v(t)
x_true = position(t)

# Euler's method and Trapezoid method with a large step size for visualization
h = 0.5
t_numeric = np.arange(t0, t_end + h, h)
x_euler = np.zeros_like(t_numeric)
x_trapezoid = np.zeros_like(t_numeric)

for i in range(1, len(t_numeric)):
    # Euler's method
    x_euler[i] = x_euler[i-1] + h * v(t_numeric[i-1])
    
    # Trapezoid method (implicit, so we use a simple fixed-point iteration)
    x_trapezoid[i] = x_trapezoid[i-1]
    for _ in range(5):  # 5 iterations should be enough for this simple problem
        x_trapezoid[i] = x_trapezoid[i-1] + h/2 * (v(t_numeric[i-1]) + v(t_numeric[i]))

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), sharex=True)

# Plot velocity function and its approximations
ax1.plot(t, v_true, 'b-', label='True velocity')
ax1.fill_between(t, 0, v_true, alpha=0.3, label='True area (displacement)')

# Add trapezoids and rectangles
for i in range(len(t_numeric) - 1):
    t_i, t_next = t_numeric[i], t_numeric[i+1]
    v_i, v_next = v(t_i), v(t_next)
    
    # Euler's rectangle (hashed pattern)
    rect = Rectangle((t_i, 0), h, v_i, fill=True, facecolor='red', edgecolor='r', alpha=0.15, hatch='///')
    ax1.add_patch(rect)
    
    # Trapezoid (dot pattern)
    trapezoid = Polygon([(t_i, 0), (t_i, v_i), (t_next, v_next), (t_next, 0)], 
                        fill=True, facecolor='green', edgecolor='g', alpha=0.15, hatch='....')
    ax1.add_patch(trapezoid)
    
    # Add area values
    euler_area = h * v_i
    trapezoid_area = h * (v_i + v_next) / 2
    ax1.text(t_i + h/2, v_i/2, f'Euler: {euler_area:.2f}', ha='center', va='bottom', color='red', fontweight='bold')
    ax1.text(t_i + h/2, (v_i + v_next)/4, f'Trapezoid: {trapezoid_area:.2f}', ha='center', va='top', color='green', fontweight='bold')

# Plot points for Euler's and Trapezoid methods
ax1.plot(t_numeric, v(t_numeric), 'ro', markersize=6, label="Euler's points")
ax1.plot(t_numeric, v(t_numeric), 'go', markersize=6, label="Trapezoid points")

ax1.set_ylabel('Velocity (m/s)', fontsize=12)
ax1.set_title("Velocity Function: True vs Numerical Approximations", fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle=':', alpha=0.7)

# Plot position function and its approximations
ax2.plot(t, x_true, 'b-', label='True position')
ax2.plot(t_numeric, x_euler, 'ro--', label="Euler's approximation", markersize=6, linewidth=2)
ax2.plot(t_numeric, x_trapezoid, 'go--', label="Trapezoid approximation", markersize=6, linewidth=2)

ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('Position (m)', fontsize=12)
ax2.set_title("Position: True vs Numerical Approximations", fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle=':', alpha=0.7)

# Add explanatory text
ax1.text(1.76, 17, "Red hashed areas: Euler's approximation\nGreen dotted areas: Trapezoid approximation", 
         bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
         fontsize=10, ha='center', va='center')

plt.tight_layout()
plt.show()