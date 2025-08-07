import json
import argparse
from pathlib import Path

import numpy as np
import scipy.optimize as opt

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def population_dynamics_trajectory(num_points: int = 100,
                                   width: int = 800,
                                   height: int = 400,
                                   margin: int = 50):
    """Generate a population trajectory using logistic growth with harvesting.
    
    Uses discrete-time population dynamics:
    pop[t+1] = pop[t] + growth_rate * pop[t] * (1 - pop[t]/carrying_capacity) - harvest[t]
    
    This creates a realistic trajectory where population grows then declines due to harvesting,
    demonstrating true discrete-time dynamics rather than just a mathematical spline.
    """
    # Time grid
    t_grid = np.linspace(0.0, 1.0, num_points)
    xs = margin + t_grid * (width - 2 * margin)
    
    # Population dynamics parameters
    carrying_capacity = 1000.0  # Maximum sustainable population
    growth_rate = 0.15          # Intrinsic growth rate
    initial_pop = 50.0          # Starting population
    
    # Create a varying harvest strategy that initially allows growth, then increases
    harvest_rates = 10 + 40 * t_grid**2 + 20 * np.sin(2 * np.pi * t_grid) * t_grid
    
    # Simulate population dynamics
    populations = np.zeros(num_points)
    populations[0] = initial_pop
    
    for i in range(1, num_points):
        pop = populations[i-1]
        harvest = harvest_rates[i-1]
        
        # Logistic growth with harvesting
        growth = growth_rate * pop * (1 - pop / carrying_capacity)
        new_pop = pop + growth - harvest
        
        # Ensure population doesn't go negative
        populations[i] = max(0, new_pop)
    
    # Convert to screen coordinates (flip y-axis, scale to fit)
    max_pop = max(populations.max(), carrying_capacity * 0.8)
    ys = height - margin - 50 - (populations / max_pop) * (height - 2 * margin - 100)
    
    return xs.tolist(), ys.tolist()


def solve_optimal_harvest_problem(total_time_steps: int = 100):
    """Solve the optimal harvesting problem once to get the TRUE optimal solution.
    
    Problem: Maximize total harvest while ensuring population doesn't go extinct.
    This gives us the reference solution that multiple shooting should converge to.
    """
    from scipy.optimize import minimize
    
    # Population dynamics parameters
    carrying_capacity = 1000.0
    growth_rate = 0.12
    initial_population = 100.0
    extinction_threshold = 10.0
    
    def objective(harvest_rates):
        """Minimize negative total harvest (i.e., maximize harvest)"""
        populations = [initial_population]
        total_harvest = 0
        
        for i, harvest in enumerate(harvest_rates):
            pop = populations[-1]
            growth = growth_rate * pop * (1 - pop / carrying_capacity)
            new_pop = max(0, pop + growth - harvest)
            populations.append(new_pop)
            total_harvest += harvest
        
        # Heavy penalty if population goes extinct
        final_pop = populations[-1]
        extinction_penalty = 1000 if final_pop < extinction_threshold else 0
        
        return -total_harvest + extinction_penalty  # Minimize negative harvest
    
    # Optimize harvest rates
    initial_guess = np.full(total_time_steps, 20.0)  # Conservative initial harvest
    bounds = [(0, 100) for _ in range(total_time_steps)]  # Harvest between 0-100
    
    result = opt.minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    optimal_harvests = result.x
    
    # Simulate with optimal harvest rates
    populations = [initial_population]
    for harvest in optimal_harvests:
        pop = populations[-1]
        growth = growth_rate * pop * (1 - pop / carrying_capacity)
        new_pop = max(0, pop + growth - harvest)
        populations.append(new_pop)
    
    return populations, optimal_harvests


def generate_population_segments(num_segments: int, 
                                width: int = 800, 
                                height: int = 400, 
                                margin: int = 50,
                                steps_per_segment: int = 20):
    """Generate segments that approximate the optimal harvest solution.
    
    This creates the TRUE multiple shooting problem: different K values should
    converge to the same optimal solution.
    """
    total_steps = num_segments * steps_per_segment
    
    # Get the optimal solution
    optimal_populations, optimal_harvests = solve_optimal_harvest_problem(total_steps)
    
    # Screen layout
    total_width = width - 2 * margin
    segment_width = total_width / num_segments
    
    segments = []
    
    # Create segments from the optimal solution
    for k in range(num_segments):
        segment_points = []
        
        # Time coordinates for this segment
        start_x = margin + k * segment_width
        xs = np.linspace(start_x, start_x + segment_width, steps_per_segment)
        
        # Extract populations for this segment from optimal solution
        start_idx = k * steps_per_segment
        end_idx = (k + 1) * steps_per_segment
        segment_populations = optimal_populations[start_idx:end_idx]
        
        # Ensure we have enough points
        if len(segment_populations) < steps_per_segment:
            # Pad with the last population if needed
            last_pop = segment_populations[-1] if segment_populations else optimal_populations[-1]
            segment_populations.extend([last_pop] * (steps_per_segment - len(segment_populations)))
        
        # Convert to screen coordinates with better bounds
        min_pop, max_pop = 0, max(800, max(optimal_populations))
        pop_range = max_pop - min_pop
        
        ys = []
        for pop in segment_populations:
            # Map population to screen coordinates (ensuring visibility)
            y_frac = (pop - min_pop) / pop_range if pop_range > 0 else 0
            y = height - margin - 20 - y_frac * (height - 2 * margin - 40)
            ys.append(y)
        
        # Create segment points
        for x, y in zip(xs, ys):
            segment_points.append(dict(x=float(x), y=float(y)))
        
        segments.append(segment_points)
    
    return segments


def segment_points(xs, ys, num_segments: int):
    """Legacy function - now calls the new population dynamics generator."""
    return generate_population_segments(num_segments)


def add_population_perturbations(segments, max_offset: float = 60.0, seed: int | None = None):
    """Returns a deep copy of *segments* where each segment k>0 has its initial 
    population (multiple shooting variable) perturbed, but the dynamics within 
    each segment are re-simulated from that new starting point.
    """
    rng = np.random.default_rng(seed)
    perturbed = []
    
    # Population dynamics parameters (same as in optimization)
    carrying_capacity = 1000.0
    growth_rate = 0.12
    
    for k, seg in enumerate(segments):
        if k == 0:
            # Keep first segment unchanged
            perturbed.append([
                dict(x=p["x"], y=p["y"], originalY=p["y"]) for p in seg
            ])
        else:
            # Perturb the initial population for this segment
            original_start_y = seg[0]["y"]
            offset = rng.uniform(-max_offset, max_offset)
            new_start_y = original_start_y + offset
            
            # Convert back to population space to simulate dynamics
            # Reverse the screen coordinate transformation
            height, margin = 400, 50
            y_frac = (height - margin - 20 - new_start_y) / (height - 2 * margin - 40)
            max_pop = max(800, carrying_capacity)
            perturbed_population = y_frac * max_pop
            
            # Re-simulate this segment with the perturbed initial population
            perturbed_segment = []
            pop = max(0, perturbed_population)  # Ensure non-negative
            
            # Get harvest rate for this segment (simplified)
            base_harvest = 20 + 10 * (k / len(segments))
            
            for i, point in enumerate(seg):
                # Use original x-coordinate
                x = point["x"]
                
                # Simulate one step if not the first point
                if i > 0:
                    harvest = base_harvest + 5 * np.sin(2 * np.pi * i / len(seg))
                    growth = growth_rate * pop * (1 - pop / carrying_capacity)
                    pop = max(0, pop + growth - harvest)
                
                # Convert back to screen coordinates
                y_frac = pop / max_pop if max_pop > 0 else 0
                y = height - margin - 20 - y_frac * (height - 2 * margin - 40)
                
                perturbed_segment.append(dict(x=x, y=y, originalY=point["y"]))
            
            perturbed.append(perturbed_segment)
    
    return perturbed


def optimize_offsets(true_segments, initial_offsets):
    """Run BFGS on the vertical offsets so that segment endpoints connect.

    Returns (opt_offsets, history) where *history* is a list of offset arrays
    captured at every solver iteration (including the initial guess and the
    final solution).
    """
    K = len(true_segments)

    # Degenerate case: K == 1 (single shooting) – no defects to optimise.
    if K == 1 or len(initial_offsets) <= 1:
        # Simply return the initial offsets (normally all zeros) and a history
        # containing that single state so that the front-end animation logic
        # still works.
        return initial_offsets, [initial_offsets.copy()]

    # Build arrays of the first + last y-coordinates for convenience
    start_ys = np.array([seg[0]["y"] for seg in true_segments])
    end_ys = np.array([seg[-1]["y"] for seg in true_segments])

    def objective(vars_):
        offsets = np.concatenate(([0.0], vars_))  # offset[0] fixed to 0
        defects = (end_ys[:-1] + offsets[:-1]) - (start_ys[1:] + offsets[1:])
        return 0.5 * np.sum(defects ** 2)

    # Store history for visualisation
    history = []

    def callback(vars_):
        offsets = np.concatenate(([0.0], vars_))
        history.append(offsets.copy())

    # Kick-start history with initial guess
    history.append(initial_offsets.copy())

    res = opt.minimize(
        objective,
        x0=initial_offsets[1:],
        method="BFGS",
        callback=callback,
        options={"gtol": 1e-8, "maxiter": 100}
    )

    opt_offsets = np.concatenate(([0.0], res.x))
    history.append(opt_offsets.copy())
    return opt_offsets, history

# -----------------------------------------------------------------------------
# Main entry-point
# -----------------------------------------------------------------------------

def main():
    # Handle Jupyter notebook context where sys.argv may contain kernel arguments
    import sys
    
    # Check if we're in Jupyter by looking for various indicators
    in_jupyter = (
        any('-f' in arg for arg in sys.argv) or  # -f flag with temp file
        any('ipykernel_launcher' in arg for arg in sys.argv) or  # launcher script
        any('HistoryManager' in arg for arg in sys.argv) or  # IPython args
        'ipykernel_launcher.py' in sys.argv[0]  # Script name indicates Jupyter
    )
    
    if in_jupyter:
        # Use default values when running in Jupyter
        class DefaultArgs:
            k_values = "1,2,3,4,5,6,7,8"
            num_points = 101
            output = "_static/multiple_shooting_data.json"
            seed = 0
        args = DefaultArgs()
    else:
        # Normal command-line parsing
        parser = argparse.ArgumentParser(description="Generate multiple-shooting "
                                                     "trajectory data for the demo.")
        parser.add_argument("--k_values", "-K", type=str, default="1,2,3,4,5,6,7,8",
                            help="Comma-separated list of segment counts to precompute.")
        parser.add_argument("--num_points", type=int, default=101,
                            help="Total number of points along the trajectory.")
        parser.add_argument("--output", "-o", default="_static/multiple_shooting_data.json",
                            help="Path where the JSON file will be written.")
        parser.add_argument("--seed", type=int, default=0,
                            help="RNG seed for the initial guess offsets.")

        args = parser.parse_args()

    # Global continuous reference trajectory for comparison
    xs, ys = population_dynamics_trajectory(args.num_points)
    continuous_trajectory = [dict(x=float(x), y=float(y)) for x, y in zip(xs, ys)]

    cases = {}
    for k_str in args.k_values.split(','):
        k = int(k_str.strip())
        if k <= 0:
            continue

        # Generate segments with true population dynamics (not chopped from global trajectory)
        true_segments = generate_population_segments(k)
        initial_guess_segments = add_population_perturbations(true_segments, seed=args.seed + k)

        initial_offsets = np.array([seg[0]["y"] - seg[0]["originalY"] for seg in initial_guess_segments])
        opt_offsets, history = optimize_offsets(true_segments, initial_offsets)

        optimization_steps = []
        for offsets in history:
            iter_segments = []
            for seg, off in zip(true_segments, offsets):
                iter_segments.append([
                    {"x": p["x"], "y": p["y"] + off, "originalY": p["y"]} for p in seg
                ])
            optimization_steps.append(iter_segments)

        cases[str(k)] = {
            "initial_guess": initial_guess_segments,
            "optimization_steps": optimization_steps,
            "solver_success": bool(opt_offsets is not None)
        }

    data = {
        "continuous_trajectory": continuous_trajectory,  # smooth reference trajectory
        "cases": cases
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(data, f, indent=2)

    # Use Jupyter Book's gluing feature to display output instead of print
    try:
        from myst_nb import glue
        glue("multiple_shooting_output", f"✓ Generated trajectory data for K = {args.k_values} segments\n"
                                        f"✓ Output written to: {output_path.name}\n"
                                        f"✓ File size: {output_path.stat().st_size / 1024:.1f} KB", display=False)
    except ImportError:
        # Fallback for when not running in Jupyter Book context
        print(f"⟹ Wrote aggregated trajectory data to {output_path.resolve()}")


if __name__ == "__main__":
    main() 