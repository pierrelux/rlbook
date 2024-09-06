import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.example_libraries import optimizers
import matplotlib.pyplot as plt

def single_shooting_ev_optimization(T=20, num_iterations=1000, step_size=0.01):
    """
    Implements the single shooting method for the electric vehicle energy optimization problem.
    
    Args:
    T: time horizon
    num_iterations: number of optimization iterations
    step_size: step size for the optimizer
    
    Returns:
    optimal_u: optimal control sequence
    """
    
    def f(x, u, t):
        return jnp.array([
            x[0] + 0.1 * x[1] + 0.05 * u,
            x[1] + 0.1 * u
        ])
    
    def c(x, u, t):
        if t == T:
            return x[0]**2 + x[1]**2
        else:
            return 0.1 * (x[0]**2 + x[1]**2 + u**2)
    
    def phi(u, x1, t):
        x = x1
        for k in range(1, t+1):
            if k > 1:
                x = f(x, u[k-2], k-1)
        return x
    
    def objective(u):
        total_cost = 0
        for t in range(1, T):
            x_t = phi(u, x1, t)
            total_cost += c(x_t, u[t-1], t)
        x_T = phi(u, x1, T)
        total_cost += c(x_T, 0.0, T)  # No control at final step
        return total_cost
    
    def clip_controls(u):
        return jnp.clip(u, -1.0, 1.0)
    
    x1 = jnp.array([1.0, 0.0])  # Initial state: full battery, zero speed
    
    # Initialize controls
    u_init = jnp.zeros(T-1)
    
    # Setup optimizer
    optimizer = optimizers.adam(step_size)
    opt_init, opt_update, get_params = optimizer
    opt_state = opt_init(u_init)
    
    @jit
    def step(i, opt_state):
        u = get_params(opt_state)
        value, grads = jax.value_and_grad(objective)(u)
        opt_state = opt_update(i, grads, opt_state)
        u = get_params(opt_state)
        u = clip_controls(u)
        opt_state = opt_init(u)
        return value, opt_state
    
    # Run optimization
    for i in range(num_iterations):
        value, opt_state = step(i, opt_state)
        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {value}")
    
    optimal_u = get_params(opt_state)
    return optimal_u

def plot_results(optimal_u, T):
    # Compute state trajectory
    x1 = jnp.array([1.0, 0.0])
    x_trajectory = [x1]
    for t in range(T-1):
        x_next = jnp.array([
            x_trajectory[-1][0] + 0.1 * x_trajectory[-1][1] + 0.05 * optimal_u[t],
            x_trajectory[-1][1] + 0.1 * optimal_u[t]
        ])
        x_trajectory.append(x_next)
    x_trajectory = jnp.array(x_trajectory)

    time = jnp.arange(T)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time, x_trajectory[:, 0], label='Battery State of Charge')
    plt.plot(time, x_trajectory[:, 1], label='Vehicle Speed')
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.title('Optimal State Trajectories')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time[:-1], optimal_u, label='Motor Power Input')
    plt.xlabel('Time Step')
    plt.ylabel('Control Input')
    plt.title('Optimal Control Inputs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Run the optimization
optimal_u = single_shooting_ev_optimization()
print("Optimal control sequence:", optimal_u)

# Plot the results
plot_results(optimal_u, T=20)