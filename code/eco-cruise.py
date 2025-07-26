import numpy as np
import json
from scipy.optimize import minimize, Bounds

def solve_eco_cruise(beta=1.0, gamma=0.05, T=60, v_max=20.0, a_max=3.0, distance=1000.0, initial_guess="triangular"):
    """
    Solve the eco-cruise optimization problem and return trajectory data.
    
    Parameters:
    - beta: acceleration penalty weight (default 1.0)
    - gamma: speed penalty weight (default 0.05, increase for more speed penalty)
    - T: time horizon (seconds)
    - v_max: maximum speed (m/s)
    - a_max: maximum acceleration magnitude (m/s^2)
    - distance: total distance to travel (m)
    - initial_guess: "triangular" (recommended) or "constant"
    
    Note: For meaningful energy savings, gamma should typically be much smaller than beta.
    Try gamma=0.01-0.1 for different energy/speed trade-offs.
    """
    
    n_state = T + 1
    n_control = T

    def unpack(z):
        s = z[:n_state]; v = z[n_state:2*n_state]; u = z[2*n_state:]
        return s, v, u

    def objective(z):
        _, v, u = unpack(z)
        return 0.5 * beta * np.sum(u**2) + 0.5 * gamma * np.sum(v[:-1]**2)

    def dynamics(z):
        s, v, u = unpack(z)
        ceq = np.empty(2*T)
        ceq[0::2] = s[1:] - s[:-1] - v[:-1]
        ceq[1::2] = v[1:] - v[:-1] - u
        return ceq

    def boundary(z):
        s, v, _ = unpack(z)
        return np.array([s[0], v[0], s[-1]-distance, v[-1]])

    # Set up optimization
    cons = [{'type':'eq', 'fun': dynamics}, {'type':'eq', 'fun': boundary}]
    big = 1e4
    lb = np.concatenate([np.full(n_state,-big), np.zeros(n_state), np.full(n_control,-a_max)])
    ub = np.concatenate([np.full(n_state, big), v_max*np.ones(n_state), np.full(n_control,a_max)])
    bounds = Bounds(lb, ub)

    # --- Initial guess ----------------------------------------------------
    constant_velocity = distance / T

    if initial_guess == "triangular":
        # Simple triangular profile: accelerate, cruise, decelerate
        accel_time = int(0.3 * T)
        decel_time = int(0.3 * T) 
        cruise_time = T - accel_time - decel_time
        
        # Ensure we have enough time for all phases
        if cruise_time < 0:
            cruise_time = 0
            accel_time = T // 2
            decel_time = T - accel_time
        
        # Target peak velocity (not too high to stay feasible)
        peak_v = min(1.2 * constant_velocity, 0.8 * v_max)
        
        # Build velocity profile with correct length (T+1)
        v0 = np.zeros(n_state)
        
        # Acceleration phase
        if accel_time > 0:
            v0[:accel_time+1] = np.linspace(0, peak_v, accel_time+1)
        
        # Cruise phase  
        if cruise_time > 0:
            v0[accel_time:accel_time+cruise_time+1] = peak_v
            
        # Deceleration phase
        if decel_time > 0:
            v0[accel_time+cruise_time:] = np.linspace(peak_v, 0, decel_time+1)
        
        # Ensure final velocity is 0
        v0[-1] = 0
        
        # Compute positions by integration
        s0 = np.zeros(n_state)
        for i in range(1, n_state):
            s0[i] = s0[i-1] + v0[i-1]
        
        # Scale to hit target distance
        if s0[-1] > 0:
            scale = distance / s0[-1]
            v0 *= scale
            s0 *= scale
        
        # Compute accelerations
        u0 = np.zeros(n_control)
        for i in range(n_control):
            u0[i] = v0[i+1] - v0[i]
            
        print(f"Triangular guess: peak_v={peak_v:.1f}, final_distance={s0[-1]:.1f}")
    else:
        # Constant-speed profile (original behaviour)
        s0 = np.linspace(0, distance, n_state)
        v0 = np.full(n_state, constant_velocity)
        u0 = np.zeros(n_control)

    z0 = np.concatenate([s0, v0, u0])

    # Solve optimization
    print("Solving eco-cruise optimization...")
    print(f"Parameters: beta={beta}, gamma={gamma}")
    print(f"Initial objective: {objective(z0):.2f}")
    
    res = minimize(
        objective,
        z0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    
    print(f"Optimization {'succeeded' if res.success else 'failed'}: {res.message}")
    print(f"Final objective: {objective(res.x):.2f}")
    
    if not res.success:
        print(f"Optimization failed: {res.message}")
        return None
        
    s_opt, v_opt, u_opt = unpack(res.x)
    
    # Calculate energy costs
    accel_costs = 0.5 * beta * u_opt**2
    speed_costs = 0.5 * gamma * v_opt[:-1]**2
    total_costs = accel_costs + speed_costs
    
    # Create trajectory data structure
    eco_trajectory = []
    cumulative_energy = 0
    
    for t in range(T + 1):
        if t < T:
            stage_cost = total_costs[t]
            cumulative_energy += stage_cost
        else:
            stage_cost = 0
            
        eco_trajectory.append({
            "time": float(t),
            "position": float(s_opt[t]),
            "velocity": float(v_opt[t]),
            "acceleration": float(u_opt[t]) if t < T else 0.0,
            "stageCost": float(stage_cost),
            "cumulativeEnergy": float(cumulative_energy)
        })
    
    return {
        "eco_trajectory": eco_trajectory,
        "total_energy": float(cumulative_energy),
        "optimization_success": True,
        "parameters": {
            "beta": beta,
            "gamma": gamma,
            "T": T,
            "v_max": v_max,
            "a_max": a_max,
            "distance": distance
        }
    }

def generate_naive_trajectory(T=60, distance=1000.0, gamma=0.1):
    """Generate the naive constant-speed trajectory for comparison."""
    
    constant_velocity = distance / T
    naive_trajectory = []
    cumulative_energy = 0
    
    for t in range(T + 1):
        position = min(constant_velocity * t, distance)
        velocity = constant_velocity if position < distance else 0
        
        # Only speed cost for naive (no acceleration changes)
        stage_cost = 0.5 * gamma * velocity**2 if t < T else 0
        cumulative_energy += stage_cost
        
        naive_trajectory.append({
            "time": float(t),
            "position": float(position), 
            "velocity": float(velocity),
            "acceleration": 0.0,
            "stageCost": float(stage_cost),
            "cumulativeEnergy": float(cumulative_energy)
        })
    
    return {
        "naive_trajectory": naive_trajectory,
        "total_energy": float(cumulative_energy)
    }

def generate_trajectory_data(filename="trajectory_data.json", **kwargs):
    """
    Generate both eco and naive trajectories and save to JSON file.
    
    Parameters:
    - filename: output JSON filename
    - **kwargs: parameters to pass to solve_eco_cruise()
    """
    
    # Solve eco-cruise optimization
    eco_data = solve_eco_cruise(**kwargs)
    
    if eco_data is None:
        print("Failed to generate eco trajectory")
        return False
    
    # Generate naive trajectory with same parameters
    T = kwargs.get('T', 60)
    distance = kwargs.get('distance', 1000.0) 
    gamma = kwargs.get('gamma', 0.1)
    
    naive_data = generate_naive_trajectory(T=T, distance=distance, gamma=gamma)
    
    # Combine data
    combined_data = {
        **eco_data,
        **naive_data,
        "energy_savings_percent": float((naive_data["total_energy"] - eco_data["total_energy"]) / naive_data["total_energy"] * 100)
    }
    
    # Save to JSON
    with open(filename, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"Trajectory data saved to {filename}")
    print(f"Eco energy: {eco_data['total_energy']:.2f}")
    print(f"Naive energy: {naive_data['total_energy']:.2f}")
    print(f"Energy savings: {combined_data['energy_savings_percent']:.1f}%")
    
    return True

def generate_multiple_scenarios():
    """Generate trajectory data for different parameter combinations."""
    
    scenarios = [
        {"beta": 1.0, "gamma": 0.1, "filename": "trajectory_default.json"},
        {"beta": 2.0, "gamma": 0.1, "filename": "trajectory_high_accel_penalty.json"},
        {"beta": 0.5, "gamma": 0.2, "filename": "trajectory_high_speed_penalty.json"},
        {"beta": 1.0, "gamma": 0.05, "filename": "trajectory_low_speed_penalty.json"},
    ]
    
    for scenario in scenarios:
        filename = scenario.pop("filename")
        print(f"\n--- Generating {filename} ---")
        generate_trajectory_data(filename=filename, **scenario)

if __name__ == "__main__":
    # Generate default trajectory
    generate_trajectory_data()
    
    # Uncomment to generate multiple scenarios
    # generate_multiple_scenarios()