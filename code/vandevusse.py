import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Define the Chemical Reactor Dynamics ---
# This section models the Van de Vusse reaction. The temperature, our control
# input, affects the reaction rates non-linearly via the Arrhenius equation.

# Reaction parameters (pre-exponential factors, activation energies)
# These values are scaled down from typical Van de Vusse parameters to work
# with shorter time scales while maintaining the challenging optimization landscape.
k10, k20, k30 = 1.287e6, 1.287e6, 9.043e3 # 1/h (scaled down)
E1, E2, E3 = 9758.3, 9758.3, 8560.0        # K
R = 1.987                                 # Universal gas constant (cal/mol/K)

def arrhenius(T_K, k0, E):
    """Calculates reaction rate k based on temperature T in Kelvin."""
    return k0 * jnp.exp(-E / (R * T_K))

# Define the system of ODEs for the reactor
def reactor_dynamics(concentrations, t, temp_K):
    """
    Defines the differential equations for the concentrations of A and B.
    
    Args:
        concentrations: A JAX array [C_A, C_B].
        t: Time (unused, but required by the ODE solver).
        temp_K: The control input, temperature in Kelvin.
        
    Returns:
        The time derivatives [dC_A/dt, dC_B/dt].
    """
    Ca, Cb = concentrations
    
    # Ensure concentrations are non-negative
    Ca = jnp.maximum(Ca, 0.0)
    Cb = jnp.maximum(Cb, 0.0)
    
    # Ensure temperature is reasonable
    temp_K = jnp.clip(temp_K, 273.15 + 20.0, 273.15 + 150.0)  # 20-150°C in Kelvin
    
    # Calculate temperature-dependent reaction rates
    k1 = arrhenius(temp_K, k10, E1)
    k2 = arrhenius(temp_K, k20, E2)
    k3 = arrhenius(temp_K, k30, E3)
    
    # Allow reaction rates to be larger - the clipping might be too conservative
    k1 = jnp.clip(k1, 0.0, 1e15)
    k2 = jnp.clip(k2, 0.0, 1e15)
    k3 = jnp.clip(k3, 0.0, 1e15)
    
    # Differential equations
    dCa_dt = -k1 * Ca - k3 * Ca**2
    dCb_dt = k1 * Ca - k2 * Cb
    
    # Allow larger derivatives - the clipping might be too conservative
    dCa_dt = jnp.clip(dCa_dt, -1e6, 1e6)
    dCb_dt = jnp.clip(dCb_dt, -1e6, 1e6)
    
    return jnp.array([dCa_dt, dCb_dt])

# --- 2. Set up the Single Shooting Problem ---
# We define a "rollout" function that simulates the entire batch process for a
# given temperature profile. The cost function is simply the negative of the
# final concentration of product B.

# Simulation parameters
t_final = 0.01  # Total batch time in hours (adjusted for scaled reaction rates)
n_steps = 100  # Number of control intervals
time_points = jnp.linspace(0, t_final, n_steps + 1)

# Initial conditions: start with pure reactant A
Ca_initial = 0.5  # mol/L
Cb_initial = 0.0  # mol/L
initial_concentrations = jnp.array([Ca_initial, Cb_initial])

def simulate_batch(temp_profile_C):
    """
    Simulates the entire reactor batch process for a given temperature profile.
    This is the "forward pass" or "rollout" in single shooting.
    
    Args:
        temp_profile_C: A JAX array of temperature values in Celsius for each interval.
        
    Returns:
        The full concentration trajectory [C_A(t), C_B(t)].
    """
    # Ensure temperature profile is finite and within bounds
    temp_profile_C = jnp.clip(temp_profile_C, 20.0, 150.0)
    temp_profile_C = jnp.nan_to_num(temp_profile_C, nan=110.0)
    
    temp_profile_K = temp_profile_C + 273.15 # Convert to Kelvin

    def dynamics_for_interval(y, t, T):
        return reactor_dynamics(y, t, T)

    # Use a scan to iterate through the control intervals
    def scan_body(y_start, i):
        # Solve the ODE for the current interval with a constant temperature
        t_interval = jnp.array([time_points[i], time_points[i+1]])
        y_end = odeint(dynamics_for_interval, y_start, t_interval, temp_profile_K[i])
        
        # Ensure concentrations are non-negative and finite
        y_end = jnp.clip(y_end, 0.0, 10.0)  # Reasonable bounds for concentrations
        y_end = jnp.nan_to_num(y_end, nan=0.0)
        
        return y_end[-1], y_end[-1] # Return final state as next start and as output

    # Run the simulation over all intervals
    _, C_traj = jax.lax.scan(scan_body, initial_concentrations, jnp.arange(n_steps))
    
    # Ensure final trajectory is finite
    C_traj = jnp.nan_to_num(C_traj, nan=0.0)
    
    # Prepend the initial state to the trajectory for a complete history
    return jnp.vstack((initial_concentrations, C_traj))

@jax.jit
def objective_function(temp_profile_C):
    """The objective is to maximize the final concentration of B."""
    full_trajectory = simulate_batch(temp_profile_C)
    final_Cb = full_trajectory[-1, 1]
    return -final_Cb  # We minimize the negative to achieve maximization

# --- 3. Run the Optimization ---
# We use Adam optimizer which is much better at escaping local minima
# and finding better solutions than simple gradient descent.

# JIT-compile the gradient function for speed
grad_fn = jax.jit(jax.grad(objective_function))

# Initial guess: Try a random profile to explore different regions
# Use a seed for reproducibility
key = jax.random.PRNGKey(42)
initial_guess = 100.0 + 50.0 * jax.random.uniform(key, (n_steps,))  # Random between 100-150°C

# Adam optimizer parameters
learning_rate = 0.5  # Even higher learning rate to push further
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
n_iterations = 500

# Initialize Adam state
m = jnp.zeros_like(initial_guess)  # First moment
v = jnp.zeros_like(initial_guess)  # Second moment
temp_profile_opt = initial_guess

# Temperature bounds for the reactor (physical limits)
T_min, T_max = 20.0, 150.0

# Conservative gradient clipping to prevent NaN values
gradient_clip_value = 5.0

print("Starting optimization with Adam...")
for i in range(n_iterations):
    grad = grad_fn(temp_profile_opt)
    
    # Check for NaN gradients and handle them
    if jnp.any(jnp.isnan(grad)):
        print(f"Warning: NaN gradient detected at iteration {i+1}")
        grad = jnp.zeros_like(grad)
    
    # Debug: Check gradient statistics
    if i == 0:  # Only print for first iteration to avoid spam
        grad_norm = jnp.linalg.norm(grad)
        grad_max = jnp.max(jnp.abs(grad))
        print(f"Initial gradient norm: {grad_norm:.6f}, max abs: {grad_max:.6f}")
    
    # Try without gradient clipping to see if it's limiting us
    # grad = jnp.clip(grad, -gradient_clip_value, gradient_clip_value)
    
    # Adam update
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    
    # Bias correction
    m_hat = m / (1 - beta1**(i + 1))
    v_hat = v / (1 - beta2**(i + 1))
    
    # Update step
    temp_profile_opt -= learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon)
    
    # Project the temperature back into the feasible range (bound constraints)
    temp_profile_opt = jnp.clip(temp_profile_opt, T_min, T_max)
    
    # Check for NaN values in the temperature profile
    if jnp.any(jnp.isnan(temp_profile_opt)):
        print(f"Warning: NaN temperature detected at iteration {i+1}")
        temp_profile_opt = jnp.clip(temp_profile_opt, T_min, T_max)
        temp_profile_opt = jnp.nan_to_num(temp_profile_opt, nan=110.0)
    
    if (i + 1) % 50 == 0:
        cost = objective_function(temp_profile_opt)
        if jnp.isnan(cost):
            print(f"Iteration {i+1:4d}, Final C_B: NaN (cost is NaN)")
        else:
            print(f"Iteration {i+1:4d}, Final C_B: {-cost:.4f}")

print("Optimization finished.")

# --- 4. Visualize the Results ---
# This is the most important part. We plot the "optimized" profile and the
# resulting concentrations. We then compare it to a known-good solution to
# show how far off our result is, highlighting the failure of single shooting
# with a naive guess.

# Simulate the final optimized trajectory
optimized_traj = simulate_batch(temp_profile_opt)
final_cb_optimized = optimized_traj[-1, 1]

# A known good (near-optimal) profile for comparison.
# This profile is non-obvious and difficult to find.
t_norm = np.linspace(0, 1, n_steps)
known_good_profile = 125 * np.exp(-1.5 * t_norm) + 25
# Convert numpy array to JAX array before passing to simulation.
known_good_traj = simulate_batch(jnp.array(known_good_profile))
final_cb_known_good = known_good_traj[-1, 1]

# Test the initial guess to see what it produces
initial_traj = simulate_batch(initial_guess)
final_cb_initial = initial_traj[-1, 1]

print(f"Initial guess produces final C_B: {final_cb_initial:.6f}")
print(f"Known good profile produces final C_B: {final_cb_known_good:.6f}")
print(f"Optimized profile produces final C_B: {final_cb_optimized:.6f}")

# Create the plots
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig.suptitle("Van de Vusse Reactor: Single Shooting with Naive Initial Guess", fontsize=16)

# Plot 1: Control Profiles (Temperature)
axs[0].plot(time_points[:-1], np.array(temp_profile_opt), 'r-', label=f'Optimized Profile (Final $C_B$ = {final_cb_optimized:.4f})')
axs[0].plot(time_points[:-1], known_good_profile, 'g--', label=f'Known Good Profile (Final $C_B$ = {final_cb_known_good:.4f})')
axs[0].set_ylabel('Temperature (°C)')
axs[0].set_title('Control Input: Temperature vs. Time')
axs[0].legend()
axs[0].grid(True)

# Plot 2: Concentration Trajectories
axs[1].plot(time_points, np.array(optimized_traj)[:, 1], 'r-', label='$C_B$ from Optimized Profile')
axs[1].plot(time_points, np.array(optimized_traj)[:, 0], 'r:', label='$C_A$ from Optimized Profile')

axs[1].plot(time_points, np.array(known_good_traj)[:, 1], 'g--', label='$C_B$ from Known Good Profile')
axs[1].plot(time_points, np.array(known_good_traj)[:, 0], 'g:', label='$C_A$ from Known Good Profile')

axs[1].set_ylabel('Concentration (mol/L)')
axs[1].set_xlabel('Time (hours)')
axs[1].set_title('State Trajectories: Concentrations vs. Time')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
