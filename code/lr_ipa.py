import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

key = jax.random.PRNGKey(0)

@jax.jit
def naive_gradient_batch(key, theta):
    samples = jax.random.normal(key, (1000,)) + theta
    return jnp.mean(samples**2)

@jax.jit
def score_function_batch(key, theta):
    samples = jax.random.normal(key, (1000,)) + theta
    return jnp.mean(samples**2 * theta * (samples - theta) + samples**2)

@jax.jit
def reparam_gradient_batch(key, theta):
    eps = jax.random.normal(key, (1000,))
    samples = theta + eps
    return jnp.mean(samples**2 + 2*theta*samples)

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