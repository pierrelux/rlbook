import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Model parameters
delta = 0.95  # discount factor
price = 1.0   # output price
C = 0.2       # replanting cost
K = 0.5       # carrying capacity
alpha = 0.1   # speed of mean reversion

# Basis functions setup
n = 350  # number of collocation nodes
snodes = np.linspace(0, K, n)  # collocation nodes (linear spacing)

# Action space (dichotomous: don't cut=0, cut=1)
actions = np.array([0, 1])


def growth_function(s):
    """Growth function for biomass: s_{t+1} = K + exp(-alpha)*(s_t - K)"""
    return K + np.exp(-alpha) * (s - K)


def reward_function(s, x):
    """
    Reward function
    x=0: don't cut, reward = 0
    x=1: cut, reward = price*s - C (profit from cutting minus replanting cost)
    """
    return (price * s - C) * x


def bellman_operator(V_interp, s):
    """
    Apply Bellman operator at state s
    Returns: max value and optimal action
    """
    # Action 0: Don't cut
    s_next_0 = growth_function(s)
    value_0 = 0 + delta * V_interp(s_next_0)
    
    # Action 1: Cut
    s_next_1 = 0  # After cutting, biomass goes to 0
    value_1 = price * s - C + delta * V_interp(s_next_1)
    
    # Choose action with maximum value
    if value_1 > value_0:
        return value_1, 1
    else:
        return value_0, 0


def solve_timber_cutting(snodes, tol=1e-8, max_iter=2000):
    """
    Solve the timber cutting model using value function iteration
    with collocation method
    """
    n = len(snodes)
    V = np.zeros(n)  # Initialize value function
    
    for iteration in range(max_iter):
        V_old = V.copy()
        
        # Create interpolator for current value function
        # Use linear interpolation with extrapolation
        V_interp = interp1d(snodes, V_old, kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
        
        # Update value function at each collocation node
        for i, s in enumerate(snodes):
            V[i], _ = bellman_operator(V_interp, s)
        
        # Check convergence
        error = np.max(np.abs(V - V_old))
        if error < tol:
            print(f"Converged in {iteration + 1} iterations with error {error:.2e}")
            break
    else:
        print(f"Warning: Did not converge after {max_iter} iterations. Error: {error:.2e}")
    
    return V


def compute_policy_and_residual(snodes, V):
    """
    Compute optimal policy and Bellman equation residuals
    
    The residual measures how well the value function satisfies the Bellman equation.
    For finite difference/linear interpolation basis, residuals show oscillatory pattern.
    """
    n = len(snodes)
    policy = np.zeros(n)
    residual = np.zeros(n)
    
    # Create interpolator for value function
    V_interp = interp1d(snodes, V, kind='linear', 
                       bounds_error=False, fill_value='extrapolate')
    
    for i, s in enumerate(snodes):
        # Compute optimal value and action from Bellman operator
        V_bellman, x_opt = bellman_operator(V_interp, s)
        policy[i] = x_opt
        
        # Bellman residual: T(V)(s) - V(s)
        # This measures the error in the Bellman equation at each node
        residual[i] = V_bellman - V[i]
    
    return policy, residual


# Solve the model
print("Solving timber cutting model...")
V = solve_timber_cutting(snodes)

# Compute policy at collocation nodes
policy, residual_nodes = compute_policy_and_residual(snodes, V)

# Create refined grid for plotting
s_refined = np.linspace(0, K, 1000)

# Use LINEAR interpolation for value function (consistent with collocation basis)
V_interp_linear = interp1d(snodes, V, kind='linear', bounds_error=False, fill_value='extrapolate')
V_refined = V_interp_linear(s_refined)

policy_interp = interp1d(snodes, policy, kind='nearest', bounds_error=False, fill_value='extrapolate')
policy_refined = policy_interp(s_refined)

# Compute residuals on refined grid
# Following the textbook: R(s) = max{f(s,x) + delta*V(g(s,x))} - V(s)
# This is T(V)(s) - V(s) where T is the Bellman operator
residual_refined = np.zeros(len(s_refined))

for i, s in enumerate(s_refined):
    # Right side: Bellman operator T(V)(s) = max over actions
    # Action 0: Don't cut
    s_next_0 = growth_function(s)
    value_0 = 0 + delta * V_interp_linear(s_next_0)
    
    # Action 1: Cut  
    s_next_1 = 0
    value_1 = price * s - C + delta * V_interp_linear(s_next_1)
    
    # T(V)(s) = max of the two action values
    TV_s = max(value_0, value_1)
    
    # Left side: V(s) from basis function approximation
    V_s = V_refined[i]
    
    # Residual: R(s) = T(V)(s) - V(s)
    # At collocation nodes, this should be zero by construction
    # Between nodes, negative residuals indicate overestimation by linear approximation
    residual_refined[i] = TV_s - V_s

# Create plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Figure 9.2a: Value Function
ax1.plot(s_refined, V_refined, 'b-', linewidth=2)
ax1.set_xlabel('Biomass (s)', fontsize=12)
ax1.set_ylabel('Value', fontsize=12)
ax1.set_title('Value Function', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, K])

# Add vertical line at cutting threshold
cut_threshold_idx = np.where(policy_refined == 1)[0]
if len(cut_threshold_idx) > 0:
    cut_threshold = s_refined[cut_threshold_idx[0]]
    ax1.axvline(cut_threshold, color='r', linestyle='--', alpha=0.5, 
                label=f'Cut threshold ≈ {cut_threshold:.3f}')
    ax1.legend()

# Figure 9.2b: Bellman Equation Residual
ax2.plot(s_refined, residual_refined, 'r-', linewidth=1.5)
ax2.set_xlabel('Biomass (s)', fontsize=12)
ax2.set_ylabel('Residual', fontsize=12)
ax2.set_title('Bellman Equation Residual', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, K])
ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

plt.tight_layout()
plt.savefig('timber_cutting_solution.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional plot: Optimal Policy
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(s_refined, policy_refined, 'g-', linewidth=2)
ax.set_xlabel('Biomass (s)', fontsize=12)
ax.set_ylabel('Action (0=Keep, 1=Cut)', fontsize=12)
ax.set_title('Optimal Policy', fontsize=14, fontweight='bold')
ax.set_ylim([-0.1, 1.1])
ax.set_xlim([0, K])
ax.grid(True, alpha=0.3)
ax.set_yticks([0, 1])
ax.set_yticklabels(['Keep', 'Cut'])

if len(cut_threshold_idx) > 0:
    ax.axvline(cut_threshold, color='r', linestyle='--', alpha=0.5,
               label=f'Cut threshold ≈ {cut_threshold:.3f}')
    ax.legend()

plt.tight_layout()
plt.savefig('timber_cutting_policy.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("TIMBER CUTTING MODEL RESULTS")
print("="*60)
print(f"Model Parameters:")
print(f"  Discount factor (δ):     {delta}")
print(f"  Price per unit (p):      {price}")
print(f"  Replanting cost (C):     {C}")
print(f"  Carrying capacity (K):   {K}")
print(f"  Growth rate (α):         {alpha}")
print(f"\nSolution:")
print(f"  Number of nodes:         {n}")
if len(cut_threshold_idx) > 0:
    print(f"  Cutting threshold:       {cut_threshold:.4f}")
print(f"  Max value:               {np.max(V):.4f}")
print(f"  Max |residual| (nodes):  {np.max(np.abs(residual_nodes)):.2e}")
print(f"  Max |residual| (refined):{np.max(np.abs(residual_refined)):.2e}")
print("="*60)