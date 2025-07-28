import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds

def solve_eco_cruise(beta=1.0, gamma=0.05, T=60, v_max=20.0, a_max=3.0, distance=1000.0):
    """Solve the eco-cruise optimization problem."""
    
    n_state, n_control = T + 1, T
    
    def unpack(z):
        s, v, u = z[:n_state], z[n_state:2*n_state], z[2*n_state:]
        return s, v, u

    def objective(z):
        _, v, u = unpack(z)
        return 0.5 * beta * np.sum(u**2) + 0.5 * gamma * np.sum(v[:-1]**2)

    def dynamics(z):
        s, v, u = unpack(z)
        ceq = np.empty(2*T)
        ceq[0::2] = s[1:] - s[:-1] - v[:-1]  # position dynamics
        ceq[1::2] = v[1:] - v[:-1] - u        # velocity dynamics
        return ceq

    def boundary(z):
        s, v, _ = unpack(z)
        return np.array([s[0], v[0], s[-1]-distance, v[-1]])  # start/end conditions

    # Optimization setup
    cons = [{'type':'eq', 'fun': dynamics}, {'type':'eq', 'fun': boundary}]
    bounds = Bounds(
        lb=np.concatenate([np.full(n_state,-1e4), np.zeros(n_state), np.full(n_control,-a_max)]),
        ub=np.concatenate([np.full(n_state,1e4), v_max*np.ones(n_state), np.full(n_control,a_max)])
    )

    # Initial guess: triangular velocity profile
    accel_time = int(0.3 * T)
    decel_time = int(0.3 * T)
    cruise_time = T - accel_time - decel_time
    peak_v = min(1.2 * distance/T, 0.8 * v_max)
    
    v0 = np.zeros(n_state)
    v0[:accel_time+1] = np.linspace(0, peak_v, accel_time+1)
    v0[accel_time:accel_time+cruise_time+1] = peak_v
    v0[accel_time+cruise_time:] = np.linspace(peak_v, 0, decel_time+1)
    
    s0 = np.cumsum(np.concatenate([[0], v0[:-1]]))
    scale = distance / s0[-1]
    s0, v0 = s0 * scale, v0 * scale
    u0 = np.diff(v0)
    
    z0 = np.concatenate([s0, v0, u0])
    
    # Solve optimization
    print(f"Solving eco-cruise optimization (β={beta}, γ={gamma})...")
    res = minimize(objective, z0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-9})
    
    if not res.success:
        print(f"Optimization failed: {res.message}")
        return None
        
    s_opt, v_opt, u_opt = unpack(res.x)
    
    # Create trajectory data
    eco_trajectory = []
    cumulative_energy = 0
    
    for t in range(T + 1):
        if t < T:
            stage_cost = 0.5 * beta * u_opt[t]**2 + 0.5 * gamma * v_opt[t]**2
            cumulative_energy += stage_cost
        else:
            stage_cost = 0
            
        eco_trajectory.append({
            "time": float(t), "position": float(s_opt[t]), "velocity": float(v_opt[t]),
            "acceleration": float(u_opt[t]) if t < T else 0.0,
            "stageCost": float(stage_cost), "cumulativeEnergy": float(cumulative_energy)
        })
    
    return {
        "eco_trajectory": eco_trajectory,
        "total_energy": float(cumulative_energy),
        "optimization_success": True,
        "parameters": {"beta": beta, "gamma": gamma, "T": T, "v_max": v_max, "a_max": a_max, "distance": distance}
    }

def generate_naive_trajectory(T=60, distance=1000.0, gamma=0.05):
    """Generate naive constant-speed trajectory for comparison."""
    
    # Simple triangular profile: accelerate, cruise, decelerate
    accel_time = decel_time = 4
    cruise_time = T - accel_time - decel_time
    cruise_speed = distance / (0.5 * accel_time + cruise_time + 0.5 * decel_time)
    
    naive_trajectory = []
    cumulative_energy = 0
    
    for t in range(T + 1):
        if t <= accel_time:
            velocity = (cruise_speed / accel_time) * t
            acceleration = cruise_speed / accel_time
        elif t <= accel_time + cruise_time:
            velocity = cruise_speed
            acceleration = 0.0
        else:
            remaining_time = T - t
            velocity = (cruise_speed / decel_time) * remaining_time
            acceleration = -cruise_speed / decel_time
        
        # Calculate position by integration
        position = 0 if t == 0 else naive_trajectory[t-1]['position'] + naive_trajectory[t-1]['velocity']
        
        # Calculate costs
        if t < T:
            stage_cost = 0.5 * 1.0 * acceleration**2 + 0.5 * gamma * velocity**2
            cumulative_energy += stage_cost
        else:
            stage_cost = 0.0
            
        naive_trajectory.append({
            "time": float(t), "position": float(position), "velocity": float(velocity),
            "acceleration": float(acceleration), "stageCost": float(stage_cost),
            "cumulativeEnergy": float(cumulative_energy)
        })
    
    return {"naive_trajectory": naive_trajectory, "total_energy": float(cumulative_energy)}

def plot_comparison(eco_data, naive_data=None, save_plot=True):
    """Create visualization plots comparing eco-cruise and naive trajectories."""
    
    eco_traj = eco_data['eco_trajectory']
    times = [p['time'] for p in eco_traj]
    positions = [p['position'] for p in eco_traj]
    velocities = [p['velocity'] for p in eco_traj]
    accelerations = [p['acceleration'] for p in eco_traj]
    energy_costs = [p['stageCost'] for p in eco_traj]
    cumulative_energy = [p['cumulativeEnergy'] for p in eco_traj]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Eco-Cruise vs Naive Trajectory Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Position vs Time
    axes[0, 0].plot(times, positions, 'b-', linewidth=2, label='Eco-Cruise')
    if naive_data:
        naive_traj = naive_data['naive_trajectory']
        naive_times = [p['time'] for p in naive_traj]
        naive_positions = [p['position'] for p in naive_traj]
        axes[0, 0].plot(naive_times, naive_positions, 'r--', linewidth=2, label='Naive')
    axes[0, 0].set_xlabel('Time (s)'); axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].set_title('Position vs Time'); axes[0, 0].grid(True, alpha=0.3); axes[0, 0].legend()
    
    # Plot 2: Velocity vs Time
    axes[0, 1].plot(times, velocities, 'b-', linewidth=2, label='Eco-Cruise')
    if naive_data:
        naive_velocities = [p['velocity'] for p in naive_traj]
        axes[0, 1].plot(naive_times, naive_velocities, 'r--', linewidth=2, label='Naive')
    axes[0, 1].set_xlabel('Time (s)'); axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Velocity vs Time'); axes[0, 1].grid(True, alpha=0.3); axes[0, 1].legend()
    
    # Plot 3: Acceleration vs Time
    axes[0, 2].plot(times[:-1], accelerations[:-1], 'b-', linewidth=2, label='Eco-Cruise')
    axes[0, 2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 2].set_xlabel('Time (s)'); axes[0, 2].set_ylabel('Acceleration (m/s²)')
    axes[0, 2].set_title('Acceleration vs Time'); axes[0, 2].grid(True, alpha=0.3); axes[0, 2].legend()
    
    # Plot 4: Stage Cost vs Time
    axes[1, 0].plot(times[:-1], energy_costs[:-1], 'b-', linewidth=2, label='Eco-Cruise')
    if naive_data:
        naive_costs = [p['stageCost'] for p in naive_traj[:-1]]
        axes[1, 0].plot(naive_times[:-1], naive_costs, 'r--', linewidth=2, label='Naive')
    axes[1, 0].set_xlabel('Time (s)'); axes[1, 0].set_ylabel('Stage Cost')
    axes[1, 0].set_title('Stage Cost vs Time'); axes[1, 0].grid(True, alpha=0.3); axes[1, 0].legend()
    
    # Plot 5: Cumulative Energy vs Time
    axes[1, 1].plot(times, cumulative_energy, 'b-', linewidth=2, label='Eco-Cruise')
    if naive_data:
        naive_cumulative = [p['cumulativeEnergy'] for p in naive_traj]
        axes[1, 1].plot(naive_times, naive_cumulative, 'r--', linewidth=2, label='Naive')
    axes[1, 1].set_xlabel('Time (s)'); axes[1, 1].set_ylabel('Cumulative Energy')
    axes[1, 1].set_title('Cumulative Energy vs Time'); axes[1, 1].grid(True, alpha=0.3); axes[1, 1].legend()
    
    # Plot 6: Phase Space (Velocity vs Position)
    axes[1, 2].plot(positions, velocities, 'b-', linewidth=2, label='Eco-Cruise')
    if naive_data:
        naive_positions = [p['position'] for p in naive_traj]
        axes[1, 2].plot(naive_positions, naive_velocities, 'r--', linewidth=2, label='Naive')
    axes[1, 2].set_xlabel('Position (m)'); axes[1, 2].set_ylabel('Velocity (m/s)')
    axes[1, 2].set_title('Phase Space: Velocity vs Position'); axes[1, 2].grid(True, alpha=0.3); axes[1, 2].legend()
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('_static/eco_cruise_visualization.png', dpi=300, bbox_inches='tight')
        print("Plot saved to _static/eco_cruise_visualization.png")
    
    plt.show()
    return fig

def demo():
    """Run complete eco-cruise demonstration with visualization."""
    
    print("=== Eco-Cruise Optimization Demo ===\n")
    
    # Solve optimization
    eco_data = solve_eco_cruise(beta=1.0, gamma=0.05, T=60, distance=1000.0)
    if eco_data is None:
        print("Optimization failed!")
        return None
    
    # Generate naive trajectory
    naive_data = generate_naive_trajectory(T=60, distance=1000.0, gamma=0.05)
    
    # Create visualization
    print("\n=== Creating Visualization ===")
    fig = plot_comparison(eco_data, naive_data, save_plot=True)
    
    # Print results
    print(f"\n=== Results ===")
    print(f"Eco-Cruise energy: {eco_data['total_energy']:.2f}")
    print(f"Naive energy: {naive_data['total_energy']:.2f}")
    energy_savings = (naive_data['total_energy'] - eco_data['total_energy']) / naive_data['total_energy'] * 100
    print(f"Energy savings: {energy_savings:.1f}%")
    
    return eco_data, naive_data, fig

if __name__ == "__main__":
    demo()