import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def v(t):
    """
    Velocity function for the ballistic trajectory.
    """
    v0 = 20   # initial velocity (m/s)
    g = 9.81  # acceleration due to gravity (m/s^2)
    return v0 - g * t

def position(t):
    """
    Position function (integral of velocity).
    """
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

# Euler's method with a large step size for visualization
h = 0.5
t_euler = np.arange(t0, t_end + h, h)
x_euler = np.zeros_like(t_euler)

for i in range(1, len(t_euler)):
    x_euler[i] = x_euler[i-1] + h * v(t_euler[i-1])

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

# Plot velocity function and its approximation
ax1.plot(t, v_true, 'b-', label='True velocity')
ax1.fill_between(t, 0, v_true, alpha=0.3, label='True area (displacement)')

# Add rectangles with hashed pattern, ruler-like annotations, and area values
for i in range(len(t_euler) - 1):
    t_i = t_euler[i]
    v_i = v(t_i)
    rect = Rectangle((t_i, 0), h, v_i, 
                     fill=True, facecolor='red', edgecolor='r', 
                     alpha=0.15, hatch='///')
    ax1.add_patch(rect)
    
    # Add ruler-like annotations
    # Vertical ruler (height)
    ax1.annotate('', xy=(t_i, 0), xytext=(t_i, v_i),
                 arrowprops=dict(arrowstyle='<->', color='red'))
    ax1.text(t_i - 0.05, v_i/2, f'v(t{i}) = {v_i:.2f}', rotation=90, 
             va='center', ha='right', color='red', fontweight='bold')
    
    # Horizontal ruler (width)
    ax1.annotate('', xy=(t_i, -1), xytext=(t_i + h, -1),
                 arrowprops=dict(arrowstyle='<->', color='red'))
    ax1.text(t_i + h/2, -2, f'h = {h}', ha='center', va='top', 
             color='red', fontweight='bold')
    
    # Add area value in the middle of each rectangle
    area = h * v_i
    ax1.text(t_i + h/2, v_i/2, f'Area = {area:.2f}', ha='center', va='center', 
             color='white', fontweight='bold', bbox=dict(facecolor='red', edgecolor='none', alpha=0.7))

# Plot only the points for Euler's method
ax1.plot(t_euler, v(t_euler), 'ro', markersize=6, label="Euler's points")
ax1.set_ylabel('Velocity (m/s)', fontsize=12)
ax1.set_title("Velocity Function and Euler's Approximation", fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.set_ylim(bottom=-3)  # Extend y-axis to show horizontal rulers

# Plot position function and its approximation
ax2.plot(t, x_true, 'b-', label='True position')
ax2.plot(t_euler, x_euler, 'ro--', label="Euler's approximation", markersize=6, linewidth=2)

# Add vertical arrows and horizontal lines to show displacement and time step
for i in range(1, len(t_euler)):
    t_i = t_euler[i]
    x_prev = x_euler[i-1]
    x_curr = x_euler[i]
    
    # Vertical line for displacement
    ax2.plot([t_i, t_i], [x_prev, x_curr], 'g:', linewidth=2)
    
    # Horizontal line for time step
    ax2.plot([t_i - h, t_i], [x_prev, x_prev], 'g:', linewidth=2)
    
    # Add text to show the displacement value
    displacement = x_curr - x_prev
    ax2.text(t_i + 0.05, (x_prev + x_curr)/2, f'+{displacement:.2f}', 
             color='green', fontweight='bold', va='center')
    
    # Add text to show the time step
    ax2.text(t_i - h/2, x_prev - 0.5, f'h = {h}', 
             color='green', fontweight='bold', ha='center', va='top')

ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('Position (m)', fontsize=12)
ax2.set_title("Position: True vs Euler's Approximation", fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle=':', alpha=0.7)

# Add explanatory text
ax1.text(1.845, 15, "Red hashed areas show\nEuler's approximation\nof the area under the curve", 
         bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
         fontsize=10, ha='center', va='center')

plt.tight_layout()
plt.show()