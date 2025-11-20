"""
Centralized plotting style configuration for the RL book.

This module provides a consistent, publication-quality style across all figures
using SciencePlots with the 'science' and 'notebook' styles, which are optimized
for the MyST-based Jupyter Book 2 ecosystem.

Usage:
    In any code cell that generates plots, add at the top:
    
    from _static.plot_style import setup_plot_style
    setup_plot_style()
    
    # Your plotting code here
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.show()
"""

import matplotlib.pyplot as plt

def setup_plot_style():
    """
    Apply the book's standard plotting style.
    
    Uses SciencePlots ['science', 'notebook'] for:
    - Clean, minimalist design suitable for web
    - Publication-quality typography
    - Colorblind-friendly default palette
    - Proper sizing for MyST book-theme
    """
    try:
        import scienceplots  # noqa: F401
        plt.style.use(['science', 'notebook'])
    except (ImportError, OSError):
        # Fallback if SciencePlots not installed or LaTeX not available
        # Apply similar styling manually
        plt.rcParams.update({
            # Figure
            'figure.dpi': 144,
            'figure.facecolor': 'white',
            'figure.autolayout': False,
            
            # Axes
            'axes.linewidth': 0.8,
            'axes.edgecolor': '#333333',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'axes.labelcolor': '#333333',
            'axes.spines.top': False,
            'axes.spines.right': False,
            
            # Grid
            'grid.color': '#b0b0b0',
            'grid.linestyle': '-',
            'grid.linewidth': 0.4,
            'grid.alpha': 0.3,
            
            # Lines
            'lines.linewidth': 1.5,
            'lines.markersize': 6,
            
            # Font
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
            'font.size': 10,
            
            # Legend
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'legend.fancybox': False,
            'legend.fontsize': 9,
            'legend.edgecolor': '#cccccc',
            
            # Ticks
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'xtick.color': '#333333',
            'ytick.color': '#333333',
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            
            # Saving
            'savefig.dpi': 144,
            'savefig.bbox': 'tight',
            'savefig.facecolor': 'white',
        })

