import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from scipy.interpolate import CubicSpline
from myst_nb import glue

# Set up the figure with subplots
fig = plt.figure(figsize=(16, 12))

# Create a 2x2 subplot layout
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

def create_collocation_plot(ax, title, node_positions, slope_nodes, endpoint_types):
    """
    Create a collocation method illustration
    
    Parameters:
    - ax: matplotlib axis
    - title: plot title
    - node_positions: list of x-positions for nodes (normalized 0-1)
    - slope_nodes: list of booleans indicating which nodes enforce slopes
    - endpoint_types: tuple of (left_type, right_type) where type is 'slope', 'eval', or 'continuity'
    """
    
    # Time interval
    t_start, t_end = 0, 1
    
    # Create a smooth trajectory curve for illustration
    x = np.linspace(t_start, t_end, 100)
    # Create an S-shaped curve to represent state trajectory
    y = 0.3 + 0.4 * np.sin(3 * np.pi * x) * np.exp(-2 * x)
    
    # Build a cubic spline of the trajectory for consistent slope evaluation
    spline = CubicSpline(x, y, bc_type='natural')
    
    # Plot the trajectory
    ax.plot(x, y, 'k-', linewidth=2, label='State trajectory x(t)')
    
    # Plot collocation points
    for i, pos in enumerate(node_positions):
        t_node = t_start + pos * (t_end - t_start)
        y_node = 0.3 + 0.4 * np.sin(3 * np.pi * t_node) * np.exp(-2 * t_node)
        
        # Endpoints are rendered in the endpoint section (as squares). Skip here.
        if np.isclose(t_node, 0.0) or np.isclose(t_node, 1.0):
            continue

        if slope_nodes[i]:
            # Blue dot for slope constraint nodes
            ax.plot(t_node, y_node, 'bo', markersize=8, markerfacecolor='blue', 
                   markeredgecolor='darkblue', linewidth=1.5)
            # Add tangent line to show slope constraint (centered on node)
            dt = 0.08  # Half-length for symmetric extension
            t_prev = max(0, t_node - dt)
            t_next = min(1, t_node + dt)
            
            # Calculate slope from the spline derivative (matches plotted curve)
            slope = spline.derivative()(t_node)
            
            # Create symmetric tangent line centered on the node
            y_prev = y_node + slope * (t_prev - t_node)
            y_next = y_node + slope * (t_next - t_node)
            ax.plot([t_prev, t_next], [y_prev, y_next], 'r--', alpha=0.8, linewidth=2)
        else:
            # Green dot for evaluation-only nodes
            ax.plot(t_node, y_node, 'go', markersize=8, markerfacecolor='lightgreen', 
                   markeredgecolor='darkgreen', linewidth=1.5)
    
    # Handle endpoints specially (always render as squares if applicable)
    endpoints = [(0, 'left'), (1, 'right')]
    for pos, side in endpoints:
        y_end = 0.3 + 0.4 * np.sin(3 * np.pi * pos) * np.exp(-2 * pos)
        end_type = endpoint_types[0] if side == 'left' else endpoint_types[1]
        
        if end_type == 'slope':
            ax.plot(pos, y_end, 'bs', markersize=10, markerfacecolor='blue', 
                   markeredgecolor='darkblue', linewidth=2)
            # Add tangent line (centered on endpoint)
            dt = 0.08  # Half-length for symmetric extension
            
            # Calculate slope from the spline derivative (matches plotted curve)
            slope = spline.derivative()(pos)
            
            t_prev = pos - dt
            t_next = pos + dt
            y_prev = y_end + slope * (t_prev - pos)
            y_next = y_end + slope * (t_next - pos)
            ax.plot([t_prev, t_next], [y_prev, y_next], 'r--', alpha=0.8, linewidth=2)
        elif end_type == 'eval':
            ax.plot(pos, y_end, 'gs', markersize=10, markerfacecolor='lightgreen', 
                   markeredgecolor='darkgreen', linewidth=2)
        elif end_type == 'continuity':
            ax.plot(pos, y_end, 'ms', markersize=10, markerfacecolor='orange', 
                   markeredgecolor='darkorange', linewidth=2)
    
    # Add time markers
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
    ax.text(0, -0.15, r'$t_k$', ha='center', va='top', fontsize=12)
    ax.text(1, -0.15, r'$t_{k+1}$', ha='center', va='top', fontsize=12)
    
    # Formatting
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 0.8)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('State', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

# Define node positions for each method (normalized to [0,1])
# Using 4 nodes total for fair comparison

# Lobatto IIIA nodes (includes both endpoints)
lobatto_nodes = [0.0, 0.276, 0.724, 1.0]
lobatto_slopes = [True, True, True, True]
lobatto_endpoints = ('slope', 'slope')

# Radau IA nodes (includes left endpoint)
radau1_nodes = [0.0, 0.155, 0.645, 0.955]
radau1_slopes = [True, True, True, True]  # Collocation at left endpoint and interior nodes
radau1_endpoints = ('slope', 'eval')  # Left: slope, Right: evaluation-only

# Radau IIA nodes (includes right endpoint)
radau2_nodes = [0.045, 0.355, 0.845, 1.0]
radau2_slopes = [True, True, True, True]  # Collocation at interior nodes and right endpoint
radau2_endpoints = ('continuity', 'slope')  # Left: continuity, Right: slope

# Gauss nodes (no endpoints)
gauss_nodes = [0.113, 0.387, 0.613, 0.887]
gauss_slopes = [True, True, True, True]
gauss_endpoints = ('eval', 'eval')

# Create subplots
ax1 = fig.add_subplot(gs[0, 0])
create_collocation_plot(ax1, 'Lobatto IIIA Method', lobatto_nodes, lobatto_slopes, lobatto_endpoints)

ax2 = fig.add_subplot(gs[0, 1])
create_collocation_plot(ax2, 'Radau IA Method', radau1_nodes, radau1_slopes, radau1_endpoints)

ax3 = fig.add_subplot(gs[1, 0])
create_collocation_plot(ax3, 'Radau IIA Method', radau2_nodes, radau2_slopes, radau2_endpoints)

ax4 = fig.add_subplot(gs[1, 1])
create_collocation_plot(ax4, 'Gauss Method', gauss_nodes, gauss_slopes, gauss_endpoints)

# Create legend
legend_elements = [
    mpatches.Patch(color='blue', label='Slope constraint (f = dynamics)'),
    mpatches.Patch(color='lightgreen', label='Polynomial evaluation only'),
    mpatches.Patch(color='orange', label='Continuity constraint'),
    plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.7, label='Tangent (slope direction)'),
    plt.Line2D([0], [0], color='black', linewidth=2, label='State trajectory')
]

fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.03), 
           ncol=3, fontsize=12, frameon=True, fancybox=True, shadow=False)

# Add main title
fig.suptitle('Collocation Methods for Optimal Control\n(Illustration of Node Types and Constraints)', 
             fontsize=16, fontweight='bold', y=0.95)

plt.tight_layout(rect=[0.04, 0.10, 0.98, 0.93])

# Glue the figure for later insertion in MyST using the glue:figure directive
glue("collocation_figure", fig, display=False)