import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def solve_docp(c_T, c_t, f_t, x_1, T, u_lb, u_ub, x_lb, x_ub):
    """
    Solve a Discrete-Time Optimal Control Problem of Bolza Type using scipy.minimize with SLSQP.
    
    Parameters:
    - c_T: function, terminal cost c_T(x_T)
    - c_t: function, stage cost c_t(x_t, u_t)
    - f_t: function, state transition f_t(x_t, u_t)
    - x_1: array, initial state
    - T: int, time horizon
    - u_lb, u_ub: arrays, lower and upper bounds for control inputs
    - x_lb, x_ub: arrays, lower and upper bounds for states
    
    Returns:
    - result: OptimizeResult object from scipy.optimize.minimize
    """
    
    n_x = len(x_1)
    n_u = len(u_lb)
    
    def objective(z):
        x = z[:T*n_x].reshape(T, n_x)
        u = z[T*n_x:].reshape(T, n_u)
        
        cost = c_T(x[-1])
        for t in range(T):
            cost += c_t(x[t], u[t])
        
        return cost
    
    def constraints(z):
        x = z[:T*n_x].reshape(T, n_x)
        u = z[T*n_x:].reshape(T, n_u)
        
        cons = []
        
        # State transition constraints
        for t in range(T-1):
            cons.extend(x[t+1] - f_t(x[t], u[t]))
        
        # Initial state constraint
        cons.extend(x[0] - x_1)
        
        return np.array(cons)
    
    # Set up bounds
    bounds = []
    for t in range(T):
        bounds.extend([(xl, xu) for xl, xu in zip(x_lb, x_ub)])
    for t in range(T):
        bounds.extend([(ul, uu) for ul, uu in zip(u_lb, u_ub)])
    
    # Initial guess
    z0 = np.zeros(T * (n_x + n_u))
    
    # Solve the optimization problem
    result = minimize(
        objective,
        z0,
        method='SLSQP',
        constraints={'type': 'eq', 'fun': constraints},
        bounds=bounds,
        options={'ftol': 1e-6, 'maxiter': 1000}
    )
    
    return result

def plot_results(x_opt, u_opt, T):
    """
    Plot the optimal states and control inputs.
    
    Parameters:
    - x_opt: array, optimal states
    - u_opt: array, optimal control inputs
    - T: int, time horizon
    """
    time = np.arange(T)
    
    plt.figure(figsize=(12, 8))
    
    # Plot states
    plt.subplot(2, 1, 1)
    plt.plot(time, x_opt[:, 0], label='Battery State of Charge')
    plt.plot(time, x_opt[:, 1], label='Vehicle Speed')
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.title('Optimal State Trajectories')
    plt.legend()
    plt.grid(True)
    
    # Plot control inputs
    plt.subplot(2, 1, 2)
    plt.plot(time, u_opt, label='Motor Power Input')
    plt.xlabel('Time Step')
    plt.ylabel('Control Input')
    plt.title('Optimal Control Inputs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def example_docp():
    # Define problem-specific functions and parameters
    def c_T(x_T):
        return x_T[0]**2 + x_T[1]**2
    
    def c_t(x_t, u_t):
        return 0.1 * (x_t[0]**2 + x_t[1]**2 + u_t[0]**2)
    
    def f_t(x_t, u_t):
        return np.array([
            x_t[0] + 0.1 * x_t[1] + 0.05 * u_t[0],
            x_t[1] + 0.1 * u_t[0]
        ])
    
    x_1 = np.array([1.0, 0.0])
    T = 20
    u_lb = np.array([-1.0])
    u_ub = np.array([1.0])
    x_lb = np.array([-5.0, -5.0])
    x_ub = np.array([5.0, 5.0])
    
    result = solve_docp(c_T, c_t, f_t, x_1, T, u_lb, u_ub, x_lb, x_ub)
    
    print("Optimization successful:", result.success)
    print("Optimal cost:", result.fun)
    
    # Extract optimal states and controls
    x_opt = result.x[:T*2].reshape(T, 2)
    u_opt = result.x[T*2:].reshape(T, 1)
    
    print("Optimal states:")
    print(x_opt)
    print("Optimal controls:")
    print(u_opt)
    
    # Plot the results
    plot_results(x_opt, u_opt, T)

if __name__ == "__main__":
    example_docp()