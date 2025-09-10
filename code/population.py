import numpy as np
import matplotlib.pyplot as plt
import json, os
from scipy.optimize import least_squares

def population_model(x, r, K):
    """
    Discrete logistic population growth model
    x_{n+1} = x_n + r * x_n * (1 - x_n/K)
    
    Parameters:
    x: current population
    r: growth rate
    K: carrying capacity
    """
    return x + r * x * (1 - x/K)

def simulate_segment(x0, r, K, steps):
    """
    Simulate population growth for a given number of steps
    """
    x = x0
    trajectory = [x]
    
    for _ in range(steps):
        x = population_model(x, r, K)
        trajectory.append(x)
    
    return np.array(trajectory)

def multiple_shooting_residual(variables, boundary_conditions, segment_length, total_time):
    """
    Residual function for multiple shooting method
    
    variables: [r, K, x1_0, x2_0, ..., xn_0] where xi_0 are initial conditions for each segment
    boundary_conditions: (x_initial, x_final)
    segment_length: number of time steps per segment
    total_time: total simulation time
    """
    x_initial, x_final = boundary_conditions
    
    # Extract parameters and initial conditions
    r = variables[0]
    K = variables[1]
    
    # Infer number of segments from variable length (parameters + segment starts)
    num_inits = len(variables) - 2  # exclude r and K
    n_segments = num_inits + 1
    segment_initials = variables[2:]
    
    residuals = []
    
    # First segment starts at the known initial condition
    current_x = x_initial
    
    for i in range(n_segments):
        if i == 0:
            # First segment uses the boundary condition
            segment_start = x_initial
        else:
            # Other segments use the variables we're solving for
            segment_start = segment_initials[i-1]
        
        # Simulate this segment
        trajectory = simulate_segment(segment_start, r, K, segment_length)
        
        # Continuity condition: end of this segment should match start of next
        if i < n_segments - 1:
            segment_end = trajectory[-1]
            next_segment_start = segment_initials[i]
            residuals.append(segment_end - next_segment_start)
        else:
            # Last segment: final value should match boundary condition
            segment_end = trajectory[-1]
            residuals.append(segment_end - x_final)
    
    return residuals

def solve_population_bvp(x_initial, x_final, total_time, n_segments):
    """
    Solve the population growth boundary value problem using multiple shooting
    """
    segment_length = total_time // n_segments
    
    # Initial guess for parameters and segment starting points
    r_guess = 0.1
    K_guess = 100.0
    segment_guesses = np.linspace(x_initial, x_final, n_segments)[1:]  # Exclude first point
    
    initial_guess = [r_guess, K_guess] + list(segment_guesses)
    
    # Track solver progress (parameter vectors at each iteration)
    _param_history = []
    def _callback(xk, *_):
        _param_history.append(xk.copy())

    # Solve the system using nonlinear least squares (allows #residuals ≠ #variables)
    result = least_squares(
        multiple_shooting_residual,
        initial_guess,
        args=((x_initial, x_final), segment_length, total_time),
        callback=_callback,
    )

    solution = result.x
    # prepend the initial guess and append final solution to obtain full history
    param_history = [np.array(initial_guess)] + _param_history + [solution]

    return solution, param_history, result.success

def plot_solution(solution, x_initial, x_final, total_time, n_segments):
    """
    Plot the complete solution
    """
    r_opt, K_opt = solution[0], solution[1]
    segment_initials = solution[2:]
    
    segment_length = total_time // n_segments
    
    # Reconstruct the full trajectory
    time_points = []
    population_values = []
    
    current_time = 0
    current_x = x_initial
    
    for i in range(n_segments):
        if i == 0:
            segment_start = x_initial
        else:
            segment_start = segment_initials[i-1]
        
        trajectory = simulate_segment(segment_start, r_opt, K_opt, segment_length)
        
        # Add time points and values for this segment
        segment_times = np.arange(current_time, current_time + segment_length + 1)
        time_points.extend(segment_times)
        population_values.extend(trajectory)
        
        current_time += segment_length
        current_x = trajectory[-1]
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot the solution
    plt.subplot(2, 1, 1)
    plt.plot(time_points, population_values, 'b-', linewidth=2, label='Multiple Shooting Solution')
    plt.scatter([0, total_time], [x_initial, x_final], color='red', s=100, 
                label='Boundary Conditions', zorder=5)
    
    # Mark segment boundaries
    for i in range(1, n_segments):
        segment_time = i * segment_length
        segment_idx = segment_time
        if segment_idx < len(population_values):
            plt.axvline(x=segment_time, color='gray', linestyle='--', alpha=0.7)
            plt.scatter(segment_time, population_values[segment_idx], 
                       color='orange', s=60, zorder=4)
    
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title(f'Population Growth Model (r={r_opt:.3f}, K={K_opt:.1f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot phase portrait
    plt.subplot(2, 1, 2)
    x_range = np.linspace(0, max(K_opt * 1.2, max(population_values) * 1.2), 1000)
    growth_rate = r_opt * x_range * (1 - x_range/K_opt)
    
    plt.plot(x_range, growth_rate, 'g-', label='Growth Rate')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.axvline(x=K_opt, color='red', linestyle='--', label=f'Carrying Capacity K={K_opt:.1f}')
    
    # Mark the trajectory points on phase portrait
    population_array = np.array(population_values[:-1])  # Exclude last point
    growth_array = r_opt * population_array * (1 - population_array/K_opt)
    plt.scatter(population_array, growth_array, c=time_points[:-1], 
               cmap='viridis', s=20, alpha=0.7)
    
    plt.xlabel('Population')
    plt.ylabel('Growth Rate')
    plt.title('Phase Portrait')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return r_opt, K_opt

def build_visualization_json(param_history, x_initial, x_final, total_time, n_segments,
                             width: int = 800, height: int = 400, margin: int = 50,
                             max_iterations: int = 10):
    """Convert optimization history into the JSON structure expected by the frontend."""
    segment_length = total_time // n_segments
    seg_width = (width - 2 * margin) / n_segments

    # Use the final carrying capacity (K) to set an upper bound for the y-mapping
    max_K = max(p[1] for p in param_history)
    max_pop = max(max_K * 1.1, x_final * 1.1)
    min_pop = 0.0

    def pop_to_y(pop: float) -> float:
        # Map population to screen Y coordinate (lower values higher on screen)
        frac = (pop - min_pop) / (max_pop - min_pop) if max_pop > min_pop else 0.0
        return height - margin - 20 - frac * (height - 2 * margin - 40)

    def variables_to_segments(vars_vec):
        r = vars_vec[0]
        K = vars_vec[1]
        seg_inits = vars_vec[2:]
        segments = []

        for seg_idx in range(n_segments):
            if seg_idx == 0:
                seg_start = x_initial
            else:
                seg_start = seg_inits[seg_idx - 1]

            traj = simulate_segment(seg_start, r, K, segment_length)
            xs = np.linspace(margin + seg_idx * seg_width,
                              margin + (seg_idx + 1) * seg_width,
                              segment_length + 1)
            seg_points = []
            for t_step, pop in enumerate(traj):
                x_coord = float(xs[t_step])
                y_coord = float(pop_to_y(pop))
                seg_points.append({"x": x_coord, "y": y_coord, "originalY": y_coord})
            segments.append(seg_points)
        return segments

    # Build initial guess and optimization frames
    param_history_clipped = param_history[: max_iterations + 1]
    initial_segments = variables_to_segments(param_history_clipped[0])
    optimization_steps = [variables_to_segments(p) for p in param_history_clipped]

    # Build continuous trajectory using final parameters
    r_final, K_final = param_history[-1][0], param_history[-1][1]
    traj = [x_initial]
    for _ in range(total_time):
        traj.append(population_model(traj[-1], r_final, K_final))

    continuous_pts = []
    for t_idx, pop in enumerate(traj):
        x_coord = float(margin + (width - 2 * margin) * t_idx / total_time)
        y_coord = float(pop_to_y(pop))
        continuous_pts.append({"x": x_coord, "y": y_coord})

    return {
        "num_segments": n_segments,
        "initial_guess": initial_segments,
        "optimization_steps": optimization_steps,
        "continuous_trajectory": continuous_pts,
    }

# --------- Batch generation for multiple K values (slider) ---------

def generate_cases_json(x_initial, x_final, total_time, ks):
    """Run the solver for each k in ks and build the multi-case JSON."""
    cases = {}
    for k in ks:
        print(f"\nGenerating case for K={k} segments…")
        sol, hist, success = solve_population_bvp(x_initial, x_final, total_time, k)
        viz = build_visualization_json(hist, x_initial, x_final, total_time, k)
        cases[str(k)] = {
            "initial_guess": viz["initial_guess"],
            "optimization_steps": viz["optimization_steps"],
            "continuous_trajectory": viz["continuous_trajectory"],
            "solver_success": bool(success),
        }
    # For convenience include continuous trajectory from the reference case (largest k)
    ref_k = ks[-1]
    _, hist_ref, _ = solve_population_bvp(x_initial, x_final, total_time, ref_k)
    continuous_traj = build_visualization_json(hist_ref, x_initial, x_final, total_time, ref_k)[
        "continuous_trajectory"
    ]
    return {"cases": cases, "continuous_trajectory": continuous_traj}


# Example usage
if __name__ == "__main__":
    # Problem setup (tunable)
    x_initial = 10.0
    x_final = 80.0
    total_time = 20

    # Generate slider cases for K = 1..8
    ks = list(range(1, 9))

    print("Generating visualization data for segment counts:", ks)
    data_json = generate_cases_json(x_initial, x_final, total_time, ks)

    # Write output JSON
    output_path = os.path.join(os.path.dirname(__file__), "..", "_static", "multiple_shooting_data.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data_json, f, indent=2)

    print(f"\nVisualization data written to {output_path}\n")
    print("Reload multiple-shooting-demo.html and move the slider!\n")

    # Additionally run and plot for a representative case (e.g., K = 4)
    n_segments = 4
    solution, _, _ = solve_population_bvp(x_initial, x_final, total_time, n_segments)
    plot_solution(solution, x_initial, x_final, total_time, n_segments)