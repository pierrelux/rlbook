import jax
import jax.numpy as jnp
from jax import grad, jit, jacfwd
import matplotlib.pyplot as plt

# Define the objective function and constraint
def f(x):
    return (x[0] - 2)**2 + (x[1] - 1)**2

def g(x):
    return x[0]**2 + x[1]**2 - 1

# Lagrangian
def L(x, lambda_):
    return f(x) + lambda_ * g(x)

# Gradient and Hessian of Lagrangian
grad_L_x = jit(grad(L, argnums=0))
grad_L_lambda = jit(grad(L, argnums=1))
hess_L_xx = jit(jacfwd(grad_L_x, argnums=0))
hess_L_xlambda = jit(jacfwd(grad_L_x, argnums=1))

# Newton's method
@jit
def newton_step(x, lambda_):
    grad_x = grad_L_x(x, lambda_)
    grad_lambda = grad_L_lambda(x, lambda_)
    hess_xx = hess_L_xx(x, lambda_)
    hess_xlambda = hess_L_xlambda(x, lambda_).reshape(-1)
    
    # Construct the full KKT matrix
    kkt_matrix = jnp.block([
        [hess_xx, hess_xlambda.reshape(-1, 1)],
        [hess_xlambda, jnp.array([[0.0]])]
    ])
    
    # Construct the right-hand side
    rhs = jnp.concatenate([-grad_x, -jnp.array([grad_lambda])])
    
    # Solve the KKT system
    delta = jnp.linalg.solve(kkt_matrix, rhs)
    
    return x + delta[:2], lambda_ + delta[2]

def solve_constrained_optimization(x0, lambda0, max_iter=100, tol=1e-6):
    x, lambda_ = x0, lambda0
    
    for i in range(max_iter):
        x_new, lambda_new = newton_step(x, lambda_)
        if jnp.linalg.norm(jnp.concatenate([x_new - x, jnp.array([lambda_new - lambda_])])) < tol:
            break
        x, lambda_ = x_new, lambda_new
    
    return x, lambda_, i+1

# Analytical solution
def analytical_solution():
    x1 = 2 / jnp.sqrt(5)
    x2 = 1 / jnp.sqrt(5)
    lambda_opt = jnp.sqrt(5) - 1
    return jnp.array([x1, x2]), lambda_opt

# Solve the problem numerically
x0 = jnp.array([0.5, 0.5])
lambda0 = 0.0
x_opt_num, lambda_opt_num, iterations = solve_constrained_optimization(x0, lambda0)

# Compute analytical solution
x_opt_ana, lambda_opt_ana = analytical_solution()

# Verify the result
print("\nNumerical Solution:")
print(f"Constraint violation: {g(x_opt_num):.6f}")
print(f"Objective function value: {f(x_opt_num):.6f}")

print("\nAnalytical Solution:")
print(f"Constraint violation: {g(x_opt_ana):.6f}")
print(f"Objective function value: {f(x_opt_ana):.6f}")

print("\nComparison:")
x_diff = jnp.linalg.norm(x_opt_num - x_opt_ana)
lambda_diff = jnp.abs(lambda_opt_num - lambda_opt_ana)
print(f"Difference in x: {x_diff}")
print(f"Difference in lambda: {lambda_diff}")

# Precision test
rtol = 1e-5  # relative tolerance
atol = 1e-8  # absolute tolerance

x_close = jnp.allclose(x_opt_num, x_opt_ana, rtol=rtol, atol=atol)
lambda_close = jnp.isclose(lambda_opt_num, lambda_opt_ana, rtol=rtol, atol=atol)

print("\nPrecision Test:")
print(f"x values are close: {x_close}")
print(f"lambda values are close: {lambda_close}")

if x_close and lambda_close:
    print("The numerical solution matches the analytical solution within the specified tolerance.")
else:
    print("The numerical solution differs from the analytical solution more than the specified tolerance.")

# Visualize the result
plt.figure(figsize=(12, 10))

# Create a mesh for the contour plot
x1_range = jnp.linspace(-1.5, 2.5, 100)
x2_range = jnp.linspace(-1.5, 2.5, 100)
X1, X2 = jnp.meshgrid(x1_range, x2_range)
Z = jnp.array([[f(jnp.array([x1, x2])) for x1 in x1_range] for x2 in x2_range])

# Plot filled contours
contour = plt.contourf(X1, X2, Z, levels=50, cmap='viridis', alpha=0.7, extent=[-1.5, 2.5, -1.5, 2.5])
plt.colorbar(contour, label='Objective Function Value')

# Plot the constraint
theta = jnp.linspace(0, 2*jnp.pi, 100)
x1 = jnp.cos(theta)
x2 = jnp.sin(theta)
plt.plot(x1, x2, color='red', linewidth=2, label='Constraint')

# Plot the optimal points (numerical and analytical) and initial point
plt.scatter(x_opt_num[0], x_opt_num[1], color='red', s=100, edgecolor='white', linewidth=2, label='Numerical Optimal Point')
plt.scatter(x_opt_ana[0], x_opt_ana[1], color='blue', s=100, edgecolor='white', linewidth=2, label='Analytical Optimal Point')
plt.scatter(x0[0], x0[1], color='green', s=100, edgecolor='white', linewidth=2, label='Initial Point')

# Add labels and title
plt.xlabel('x1', fontsize=12)
plt.ylabel('x2', fontsize=12)
plt.title('Constrained Optimization: Numerical vs Analytical Solution', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Set the axis limits explicitly
plt.xlim(-1.5, 2.5)
plt.ylim(-1.5, 2.5)

plt.tight_layout()
plt.show()

