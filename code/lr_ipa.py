import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

key = jax.random.PRNGKey(0)

# Define the objective function f(x,θ) = x²θ where x ~ N(θ, 1)
def objective(x, theta):
    return x**2 * theta

# Naive Monte Carlo gradient estimation
@jax.jit
def naive_gradient_batch(key, theta):
    samples = jax.random.normal(key, (1000,)) + theta
    # Use jax.grad on the objective with respect to theta
    grad_fn = jax.grad(lambda t: jnp.mean(objective(samples, t)))
    return grad_fn(theta)

# Score function estimator (REINFORCE)
@jax.jit
def score_function_batch(key, theta):
    samples = jax.random.normal(key, (1000,)) + theta
    # f(x,θ) * ∂logp(x|θ)/∂θ + ∂f(x,θ)/∂θ
    # score function for N(θ,1) is (x-θ)
    score = samples - theta
    return jnp.mean(objective(samples, theta) * score + samples**2)

# Reparameterization gradient
@jax.jit
def reparam_gradient_batch(key, theta):
    eps = jax.random.normal(key, (1000,))
    # Use reparameterization x = θ + ε, ε ~ N(0,1)
    grad_fn = jax.grad(lambda t: jnp.mean(objective(t + eps, t)))
    return grad_fn(theta)

# Run trials
n_trials = 1000
theta = 1.0
true_grad = 3 + theta**2

keys = jax.random.split(key, n_trials)
naive_estimates = jnp.array([naive_gradient_batch(k, theta) for k in keys])
score_estimates = jnp.array([score_function_batch(k, theta) for k in keys])
reparam_estimates = jnp.array([reparam_gradient_batch(k, theta) for k in keys])

# Create violin plots with individual points
plt.figure(figsize=(12, 6))
data = [naive_estimates, score_estimates, reparam_estimates]
colors = ['#ff9999', '#66b3ff', '#99ff99']

parts = plt.violinplot(data, showextrema=False)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)

# Add box plots
plt.boxplot(data, notch=True, showfliers=False)

# Add true gradient line
plt.axhline(y=true_grad, color='r', linestyle='--', label='True Gradient')

plt.xticks([1, 2, 3], ['Naive', 'Score Function', 'Reparam'])
plt.ylabel('Gradient Estimate')
plt.title(f'Gradient Estimators (θ={theta}, true grad={true_grad:.2f})')
plt.grid(True, alpha=0.3)
plt.legend()

# Print statistics
methods = {
    'Naive': naive_estimates,
    'Score Function': score_estimates, 
    'Reparameterization': reparam_estimates
}

for name, estimates in methods.items():
    bias = jnp.mean(estimates) - true_grad
    variance = jnp.var(estimates)
    print(f"\n{name}:")
    print(f"Mean: {jnp.mean(estimates):.6f}")
    print(f"Bias: {bias:.6f}")
    print(f"Variance: {variance:.6f}")
    print(f"MSE: {bias**2 + variance:.6f}")