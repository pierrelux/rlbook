---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Gradient Estimation for Stochastic Objectives

The previous chapter differentiated a deterministic actor through a learned
critic. Stochastic policies and stochastic objectives add a sampled variable
between the parameters and the outcome. How can the derivative of an
expectation be estimated from those samples?

Two constructions answer the question. Score-function estimators differentiate
the log density of the sample, while reparameterization differentiates a
sampled outcome written as a function of parameter-free noise.

## Learning Goals

After reading this chapter, you should be able to:

- derive score-function and reparameterization estimators for a stochastic
  objective;
- identify the assumptions that make each estimator unbiased;
- compare estimator bias and variance in a controlled numerical experiment.

## Prerequisites

The chapter assumes familiarity with expectations, probability densities, and
multivariate differentiation. [Monte Carlo Bellman
Estimation](monte-carlo-bellman-estimation.md) reviews sample averages and their
variance.

## Derivative Estimation for Stochastic Optimization

When an objective averages over a parameter-dependent distribution, can its
gradient be moved inside the expectation without differentiating the sampling
operation directly?

Consider optimizing an objective that involves an expectation:

$$
J(\theta) = \mathbb{E}_{x \sim p(x;\theta)}[f(x,\theta)]
$$

For concreteness, consider a simple example where $x \sim \mathcal{N}(\theta,1)$ and $f(x,\theta) = x^2\theta$. The derivative we seek is:

$$
\frac{d}{d\theta}J(\theta) = \frac{d}{d\theta}\int x^2\theta p(x;\theta)dx
$$

While we can compute this exactly for the Gaussian example, this is often impossible for more general problems. We might then be tempted to approximate our objective using samples:

$$
J(\theta) \approx \frac{1}{N}\sum_{i=1}^N f(x_i,\theta), \quad x_i \sim p(x;\theta)
$$

Then differentiate this approximation:

$$
\frac{d}{d\theta}J(\theta) \approx \frac{1}{N}\sum_{i=1}^N \frac{\partial}{\partial \theta}f(x_i,\theta)
$$

However, this naive approach ignores that the samples themselves depend on $\theta$. The correct derivative requires the product rule:

$$
\frac{d}{d\theta}J(\theta) = \int \frac{\partial}{\partial \theta}[f(x,\theta)p(x;\theta)]dx = \int \left[\frac{\partial f}{\partial \theta}p(x;\theta) + f(x,\theta)\frac{\partial p(x;\theta)}{\partial \theta}\right]dx
$$

While the first term could be numerically integrated using Monte Carlo, the second one cannot as it is not in the form of an expectation. 

To transform our objective so that the Monte Carlo estimator for the objective could be differentiated directly while ensuring that the resulting derivative is unbiased, there are two main solutions: a change of measure, or a change of variables. 

### The Likelihood Ratio Method

One solution comes from rewriting our objective using a proposal distribution $q(x)$ that does not depend on $\theta$:

$$
J(\theta) = \int f(x,\theta)\frac{p(x;\theta)}{q(x)}q(x)dx = \mathbb{E}_{x \sim q(x)}\left[f(x,\theta)\frac{p(x;\theta)}{q(x)}\right]
$$

Define the likelihood ratio $\rho(x, q, \theta) \equiv \frac{p(x;\theta)}{q(x)}$, where we treat $q$ as a separate argument. The objective becomes:

$$
J(\theta) = \mathbb{E}_{x \sim q(x)}[f(x,\theta)\rho(x, q, \theta)]
$$

When we differentiate $J$, we take the partial derivative with respect to $\theta$ while holding $q$ fixed (since $q$ does not depend on $\theta$):

$$
\frac{d}{d\theta}J(\theta) = \mathbb{E}_{x \sim q(x)}\left[f(x,\theta)\frac{\partial \rho}{\partial \theta}(x, q, \theta) + \rho(x, q, \theta)\frac{\partial f}{\partial \theta}(x,\theta)\right]
$$

The partial derivative of $\rho$ with respect to $\theta$ (treating $q$ as fixed) is:

$$
\frac{\partial \rho}{\partial \theta}(x, q, \theta) = \frac{1}{q(x)}\frac{\partial p(x;\theta)}{\partial \theta} = \rho(x, q, \theta)\frac{\partial \log p(x;\theta)}{\partial \theta}
$$

Now fix any reference parameter $\theta_0$ and choose the proposal distribution $q(x) = p(x;\theta_0)$. This is a *fixed* distribution that does not change as $\theta$ varies. We simply evaluate the family $p(x;\cdot)$ at the specific point $\theta_0$. With this choice, evaluating the gradient at $\theta = \theta_0$ gives $\rho(x, q, \theta_0) = p(x;\theta_0)/p(x;\theta_0) = 1$. The gradient formula becomes:

$$
\frac{d}{d\theta}J(\theta)\Big|_{\theta=\theta_0} = \mathbb{E}_{x \sim p(x;\theta_0)}\left[f(x,\theta_0)\frac{\partial \log p(x;\theta)}{\partial \theta}\Big|_{\theta_0} + \frac{\partial f(x,\theta)}{\partial \theta}\Big|_{\theta_0}\right]
$$

Since $\theta_0$ is arbitrary, we can drop the subscript and write the **score function estimator** as:

$$
\frac{d}{d\theta}J(\theta) = \mathbb{E}_{x \sim p(x;\theta)}\left[f(x,\theta)\frac{\partial \log p(x;\theta)}{\partial \theta} + \frac{\partial f(x,\theta)}{\partial \theta}\right]
$$


### The Reparameterization Trick

An alternative approach eliminates the $\theta$-dependence in the sampling distribution by expressing $x$ through a deterministic transformation of the noise:

$$
x = g(\epsilon,\theta), \quad \epsilon \sim q(\epsilon)
$$

Therefore if we want to sample from some target distribution $p(x;\theta)$, we can do so by first sampling from a simple base distribution $q(\epsilon)$ (like a standard normal) and then transforming those samples through a carefully chosen function $g$. If $g(\cdot,\theta)$ is invertible, the change of variables formula tells us how these distributions relate:

$$
p(x;\theta) = q(g^{-1}(x,\theta))\left|\det\frac{\partial g^{-1}(x,\theta)}{\partial x}\right| = q(\epsilon)\left|\det\frac{\partial g(\epsilon,\theta)}{\partial \epsilon}\right|^{-1}
$$


For example, if we want to sample from any multivariate Gaussian distributions with covariance matrix $\Sigma$ and mean $\mu$, it suffices to be able to sample from a standard normal noise and compute the linear transformation:

$$
x = \mu + \Sigma^{1/2}\epsilon, \quad \epsilon \sim \mathcal{N}(0,I)
$$

where $\Sigma^{1/2}$ is the matrix square root obtained via Cholesky decomposition. In the univariate case, this transformation is simply: 

$$
x = \mu + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
$$

where $\sigma = \sqrt{\sigma^2}$ is the standard deviation (square root of the variance).



#### Common Examples of Reparameterization

##### The Truncated Normal Distribution
When we need samples constrained to an interval $[a,b]$, we can use the truncated normal distribution. To sample from it, we transform uniform noise through the inverse cumulative distribution function (CDF) of the standard normal:

$$
x = \Phi^{-1}(u\Phi(b) + (1-u)\Phi(a)), \quad u \sim \text{Uniform}(0,1)
$$

Here:
- $\Phi(z) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{z}{\sqrt{2}}\right)\right]$ is the CDF of the standard normal distribution
- $\Phi^{-1}$ is its inverse (the quantile function)
- $\text{erf}(z) = \frac{2}{\sqrt{\pi}}\int_0^z e^{-t^2}dt$ is the error function

The resulting samples follow a normal distribution restricted to $[a,b]$, with the density properly normalized over this interval.

##### The Kumaraswamy Distribution
When we need samples in the unit interval [0,1], a natural choice might be the Beta distribution. However, its inverse CDF doesn't have a closed form. Instead, we can use the Kumaraswamy distribution as a convenient approximation, which allows for a simple reparameterization:

$$
x = (1-(1-u^{\alpha})^{1/\beta}), \quad u \sim \text{Uniform}(0,1)
$$

where:
- $\alpha, \beta > 0$ are shape parameters that control the distribution
- $\alpha$ determines the concentration around 0 
- $\beta$ determines the concentration around 1
- The distribution is similar to Beta(α,β) but with analytically tractable CDF and inverse CDF

The Kumaraswamy distribution has density:

$$
f(x; \alpha, \beta) = \alpha\beta x^{\alpha-1}(1-x^{\alpha})^{\beta-1}, \quad x \in [0,1]
$$

##### The Gumbel-Softmax Distribution 

When sampling from a categorical distribution with probabilities $\{\pi_i\}$, one approach uses $\text{Gumbel}(0,1)$ noise combined with the argmax of log-perturbed probabilities:

$$
\text{argmax}_i(\log \pi_i + g_i), \quad g_i \sim \text{Gumbel}(0,1)
$$

This approach, known in machine learning as the Gumbel-Max trick, relies on sampling Gumbel noise from uniform random variables through the transformation $g_i = -\log(-\log(u_i))$ where $u_i \sim \text{Uniform}(0,1)$. To see why this gives us samples from the categorical distribution, consider the probability of selecting category $i$:

$$
\begin{align*}
P(\text{argmax}_j(\log \pi_j + g_j) = i) &= P(\log \pi_i + g_i > \log \pi_j + g_j \text{ for all } j \neq i) \\
&= P(g_i - g_j > \log \pi_j - \log \pi_i \text{ for all } j \neq i)
\end{align*}
$$

Since the difference of two Gumbel random variables follows a logistic distribution, $g_i - g_j \sim \text{Logistic}(0,1)$, and these differences are independent for different $j$ (due to the independence of the original Gumbel variables), we can write:

$$
\begin{align*}
P(\text{argmax}_j(\log \pi_j + g_j) = i) &= \prod_{j \neq i} P(g_i - g_j > \log \pi_j - \log \pi_i) \\
&= \prod_{j \neq i} \frac{\pi_i}{\pi_i + \pi_j} = \pi_i
\end{align*}
$$

The last equality requires some additional algebra to show, but follows from the fact that these probabilities must sum to 1 over all $i$.

While we have shown that the Gumbel-Max trick gives us exact samples from a categorical distribution, the argmax operation isn't differentiable. For stochastic optimization problems of the form:

$$
\mathbb{E}_{x \sim p(x;\theta)}[f(x)] = \mathbb{E}_{\epsilon \sim \text{Gumbel}(0,1)}[f(g(\epsilon,\theta))]
$$

we need $g$ to be differentiable with respect to $\theta$. This leads us to consider a continuous relaxation where we replace the hard argmax with a temperature-controlled softmax:

$$
z_i = \frac{\exp((\log \pi_i + g_i)/\tau)}{\sum_j \exp((\log \pi_j + g_j)/\tau)}
$$

As $\tau \to 0$, this approximation approaches the argmax:

$$
\lim_{\tau \to 0} \frac{\exp(x_i/\tau)}{\sum_j \exp(x_j/\tau)} = \begin{cases} 1 & \text{if } x_i = \max_j x_j \\ 0 & \text{otherwise} \end{cases}
$$

The resulting distribution over the probability simplex is called the Gumbel-Softmax (or Concrete) distribution. The temperature parameter $\tau$ controls the discreteness of our samples: smaller values give samples closer to one-hot vectors but with less stable gradients, while larger values give smoother gradients but more diffuse samples.


### Numerical Analysis of Gradient Estimators

Let us examine the behavior of our three gradient estimators for the stochastic optimization objective: 

$$J(\theta) = \mathbb{E}_{x \sim \mathcal{N}(\theta,1)}[x^2\theta]$$ 

To get an analytical expression for the derivative, first note that we can factor out $\theta$ to obtain $J(\theta) = \theta\mathbb{E}[x^2]$ where $x \sim \mathcal{N}(\theta,1)$. By definition of the variance, we know that $\text{Var}(x) = \mathbb{E}[x^2] - (\mathbb{E}[x])^2$, which we can rearrange to $\mathbb{E}[x^2] = \text{Var}(x) + (\mathbb{E}[x])^2$. Since $x \sim \mathcal{N}(\theta,1)$, we have $\text{Var}(x) = 1$ and $\mathbb{E}[x] = \theta$, therefore $\mathbb{E}[x^2] = 1 + \theta^2$. This gives us:

$$J(\theta) = \theta(1 + \theta^2)$$

Now differentiating with respect to $\theta$ using the product rule yields:

$$\frac{d}{d\theta}J(\theta) = 1 + 3\theta^2$$ 

For concreteness, we fix $\theta = 1.0$ and analyze samples drawn using Monte Carlo estimation with batch size 1000 and 1000 independent trials. Evaluating at $\theta = 1$ gives us $\frac{d}{d\theta}J(\theta)\big|_{\theta=1} = 1 + 3(1)^2 = 4$, which serves as our ground truth against which we compare our estimators:

1.  First, we consider the naive estimator that incorrectly differentiates the Monte Carlo approximation:

    $$\hat{g}_{\text{naive}}(\theta) = \frac{1}{N}\sum_{i=1}^N x_i^2$$

    For $x \sim \mathcal{N}(1,1)$, we have $\mathbb{E}[x^2] = \theta^2 + 1 = 2.0$ and $\mathbb{E}[\hat{g}_{\text{naive}}] = 2.0$. We should therefore expect a bias of about $-2$ in our experiment. 

2. Then we compute the score function estimator:

    $$\hat{g}_{\text{SF}}(\theta) = \frac{1}{N}\sum_{i=1}^N \left[x_i^2\theta(x_i - \theta) + x_i^2\right]$$

    This estimator is unbiased with $\mathbb{E}[\hat{g}_{\text{SF}}] = 4$

3. Finally, through the reparameterization $x = \theta + \epsilon$ where $\epsilon \sim \mathcal{N}(0,1)$, we obtain:

    $$\hat{g}_{\text{RT}}(\theta) = \frac{1}{N}\sum_{i=1}^N \left[2\theta(\theta + \epsilon_i) + (\theta + \epsilon_i)^2\right]$$

    This estimator is also unbiased with $\mathbb{E}[\hat{g}_{\text{RT}}] = 4$.


```{code-cell} python
:tags: [hide-input]

%config InlineBackend.figure_format = 'retina'
import jax
import jax.numpy as jnp
import altair as alt
import numpy as np
import pandas as pd

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
true_grad = 1 + 3 * theta**2

keys = jax.random.split(key, n_trials)
naive_estimates = jnp.array([naive_gradient_batch(k, theta) for k in keys])
score_estimates = jnp.array([score_function_batch(k, theta) for k in keys])
reparam_estimates = jnp.array([reparam_gradient_batch(k, theta) for k in keys])

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

gradient_data = pd.concat(
    [
        pd.DataFrame({
            "Estimator": name,
            "Gradient estimate": np.asarray(estimates),
        })
        for name, estimates in methods.items()
    ],
    ignore_index=True,
)
estimator_pick = alt.selection_point(fields=["Estimator"], bind="legend")

density = (
    alt.Chart(gradient_data)
    .transform_density(
        "Gradient estimate",
        as_=["Gradient estimate", "Density"],
        groupby=["Estimator"],
    )
    .mark_area(opacity=0.45, line=True)
    .encode(
        x=alt.X("Gradient estimate:Q", title="Gradient estimate"),
        y=alt.Y("Density:Q", stack=None),
        color=alt.Color("Estimator:N", legend=alt.Legend(orient="top")),
        opacity=alt.condition(estimator_pick, alt.value(0.55), alt.value(0.08)),
        tooltip=["Estimator:N"],
    )
    .add_params(estimator_pick)
)

truth = (
    alt.Chart(pd.DataFrame({"True gradient": [true_grad]}))
    .mark_rule(color="#b91c1c", strokeDash=[6, 4], size=2)
    .encode(x="True gradient:Q")
)

(density + truth).properties(
    height=340,
    title=f"Gradient estimator distributions (θ={theta}, true gradient={true_grad:.2f})",
)

```

The numerical experiments corroborate our theory. The naive estimator consistently underestimates the true gradient by 2.0, though it maintains a relatively small variance. This systematic bias would make it unsuitable for optimization despite its low variance. The score function estimator corrects this bias but introduces substantial variance. While unbiased, this estimator would require many samples to achieve reliable gradient estimates. Finally, the reparameterization trick achieves a much lower variance while remaining unbiased. While this experiment is for didactic purposes only, it reproduces what is commonly found in practice: that when applicable, the reparameterization estimator tends to perform better than the score function counterpart.

## Summary and Outlook

The score-function identity differentiates a log density and applies even when
the sampled variable is discrete, but its variance can be large.
Reparameterization differentiates the sampled outcome with respect to its
parameters and usually has lower variance, but it requires a differentiable
sampling path. The numerical comparison separates those variance and bias
properties directly.

Entropy-regularized continuous control requires both sampled actions and a
learned value signal. Can these estimators train a stochastic actor while
avoiding an intractable integral over actions? [Regularized and residual-based
policy learning](regularized-policy-learning.md) develops SAC, path consistency,
and related constructions.

## Self-checks

:::{exercise} Reparameterization boundary
:label: ex-pg-check-3

Give one setting where the score-function method applies but a pathwise reparameterization gradient is not directly available.
:::

:::{solution} ex-pg-check-3
:class: dropdown

A policy with discrete actions is the standard example: its samples are not differentiable functions of continuous noise, while their log probabilities remain differentiable in the policy parameters.
:::
