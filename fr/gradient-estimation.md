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
# Estimation de gradient pour des objectifs stochastiques

Le chapitre précédent a dérivé un acteur déterministe à travers un critique
appris. Les politiques stochastiques et les objectifs stochastiques ajoutent une
variable échantillonnée entre les paramètres et le résultat. Comment estimer la
dérivée d'une espérance à partir de ces échantillons?

Deux constructions répondent à la question. Les estimateurs par fonction de
score dérivent le logarithme de la densité de l'échantillon, tandis que la
reparamétrisation dérive un résultat échantillonné écrit comme une fonction d'un
bruit indépendant des paramètres.

## Objectifs d'apprentissage

Après l'étude de ce chapitre, vous devriez pouvoir :

- dériver les estimateurs par fonction de score et par reparamétrisation pour
  un objectif stochastique;
- déterminer les hypothèses qui rendent chaque estimateur sans biais;
- comparer le biais et la variance des estimateurs dans une expérience
  numérique contrôlée.

## Préalables

Le chapitre suppose une familiarité avec les espérances, les densités de
probabilité et la dérivation à plusieurs variables. [L'estimation de Bellman par
Monte-Carlo](monte-carlo-bellman-estimation.md) revoit les moyennes empiriques
et leur variance.

## Estimation de dérivées pour l'optimisation stochastique

Lorsqu'un objectif fait la moyenne sur une distribution qui dépend d'un
paramètre, peut-on faire passer son gradient à l'intérieur de l'espérance sans
dériver directement l'opération d'échantillonnage?

Considérons l'optimisation d'un objectif qui fait intervenir une espérance :

$$
J(\theta) = \mathbb{E}_{x \sim p(x;\theta)}[f(x,\theta)]
$$

Pour fixer les idées, prenons un exemple simple où $x \sim \mathcal{N}(\theta,1)$ et $f(x,\theta) = x^2\theta$. La dérivée cherchée est :

$$
\frac{d}{d\theta}J(\theta) = \frac{d}{d\theta}\int x^2\theta p(x;\theta)dx
$$

On peut la calculer exactement pour l'exemple gaussien, mais c'est souvent impossible pour des problèmes plus généraux. On pourrait alors être tenté d'approcher l'objectif à l'aide d'échantillons :

$$
J(\theta) \approx \frac{1}{N}\sum_{i=1}^N f(x_i,\theta), \quad x_i \sim p(x;\theta)
$$

puis de dériver cette approximation :

$$
\frac{d}{d\theta}J(\theta) \approx \frac{1}{N}\sum_{i=1}^N \frac{\partial}{\partial \theta}f(x_i,\theta)
$$

Cette approche naïve ignore cependant que les échantillons eux-mêmes dépendent de $\theta$. La dérivée correcte exige la règle du produit :

$$
\frac{d}{d\theta}J(\theta) = \int \frac{\partial}{\partial \theta}[f(x,\theta)p(x;\theta)]dx = \int \left[\frac{\partial f}{\partial \theta}p(x;\theta) + f(x,\theta)\frac{\partial p(x;\theta)}{\partial \theta}\right]dx
$$

Le premier terme pourrait être intégré numériquement par Monte-Carlo, mais pas le second, car il n'a pas la forme d'une espérance.

Il existe deux façons principales de transformer l'objectif afin que l'estimateur de Monte-Carlo de l'objectif puisse être dérivé directement tout en garantissant que la dérivée obtenue est sans biais : un changement de mesure ou un changement de variables.

### La méthode du rapport de vraisemblance

Une solution consiste à réécrire l'objectif à l'aide d'une loi de proposition $q(x)$ qui ne dépend pas de $\theta$ :

$$
J(\theta) = \int f(x,\theta)\frac{p(x;\theta)}{q(x)}q(x)dx = \mathbb{E}_{x \sim q(x)}\left[f(x,\theta)\frac{p(x;\theta)}{q(x)}\right]
$$

Définissons le rapport de vraisemblance $\rho(x, q, \theta) \equiv \frac{p(x;\theta)}{q(x)}$, où l'on traite $q$ comme un argument distinct. L'objectif devient :

$$
J(\theta) = \mathbb{E}_{x \sim q(x)}[f(x,\theta)\rho(x, q, \theta)]
$$

Pour dériver $J$, on prend la dérivée partielle par rapport à $\theta$ en gardant $q$ fixe (puisque $q$ ne dépend pas de $\theta$) :

$$
\frac{d}{d\theta}J(\theta) = \mathbb{E}_{x \sim q(x)}\left[f(x,\theta)\frac{\partial \rho}{\partial \theta}(x, q, \theta) + \rho(x, q, \theta)\frac{\partial f}{\partial \theta}(x,\theta)\right]
$$

La dérivée partielle de $\rho$ par rapport à $\theta$ (avec $q$ fixe) est :

$$
\frac{\partial \rho}{\partial \theta}(x, q, \theta) = \frac{1}{q(x)}\frac{\partial p(x;\theta)}{\partial \theta} = \rho(x, q, \theta)\frac{\partial \log p(x;\theta)}{\partial \theta}
$$

Fixons maintenant un paramètre de référence quelconque $\theta_0$ et choisissons la loi de proposition $q(x) = p(x;\theta_0)$. C'est une distribution *fixe*, qui ne change pas lorsque $\theta$ varie : on évalue simplement la famille $p(x;\cdot)$ au point particulier $\theta_0$. Avec ce choix, l'évaluation du gradient en $\theta = \theta_0$ donne $\rho(x, q, \theta_0) = p(x;\theta_0)/p(x;\theta_0) = 1$. La formule du gradient devient :

$$
\frac{d}{d\theta}J(\theta)\Big|_{\theta=\theta_0} = \mathbb{E}_{x \sim p(x;\theta_0)}\left[f(x,\theta_0)\frac{\partial \log p(x;\theta)}{\partial \theta}\Big|_{\theta_0} + \frac{\partial f(x,\theta)}{\partial \theta}\Big|_{\theta_0}\right]
$$

Comme $\theta_0$ est arbitraire, on peut omettre l'indice et écrire l'**estimateur par fonction de score** :

$$
\frac{d}{d\theta}J(\theta) = \mathbb{E}_{x \sim p(x;\theta)}\left[f(x,\theta)\frac{\partial \log p(x;\theta)}{\partial \theta} + \frac{\partial f(x,\theta)}{\partial \theta}\right]
$$


### L'astuce de reparamétrisation

Une autre approche élimine la dépendance en $\theta$ de la distribution d'échantillonnage en exprimant $x$ au moyen d'une transformation déterministe du bruit :

$$
x = g(\epsilon,\theta), \quad \epsilon \sim q(\epsilon)
$$

Ainsi, pour échantillonner une distribution cible $p(x;\theta)$, on peut d'abord échantillonner une distribution de base simple $q(\epsilon)$ (comme une loi normale centrée réduite), puis transformer ces échantillons au moyen d'une fonction $g$ choisie avec soin. Si $g(\cdot,\theta)$ est inversible, la formule du changement de variables indique comment ces distributions sont reliées :

$$
p(x;\theta) = q(g^{-1}(x,\theta))\left|\det\frac{\partial g^{-1}(x,\theta)}{\partial x}\right| = q(\epsilon)\left|\det\frac{\partial g(\epsilon,\theta)}{\partial \epsilon}\right|^{-1}
$$


Par exemple, pour échantillonner une loi gaussienne multivariée quelconque de matrice de covariance $\Sigma$ et de moyenne $\mu$, il suffit de pouvoir échantillonner un bruit normal centré réduit et de calculer la transformation linéaire :

$$
x = \mu + \Sigma^{1/2}\epsilon, \quad \epsilon \sim \mathcal{N}(0,I)
$$

où $\Sigma^{1/2}$ est la racine carrée matricielle obtenue par la décomposition de Cholesky. Dans le cas univarié, cette transformation est simplement :

$$
x = \mu + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
$$

où $\sigma = \sqrt{\sigma^2}$ est l'écart-type (la racine carrée de la variance).



#### Exemples courants de reparamétrisation

##### La loi normale tronquée
Lorsqu'il faut des échantillons contraints à un intervalle $[a,b]$, on peut utiliser la loi normale tronquée. Pour l'échantillonner, on transforme un bruit uniforme au moyen de l'inverse de la fonction de répartition de la loi normale centrée réduite :

$$
x = \Phi^{-1}(u\Phi(b) + (1-u)\Phi(a)), \quad u \sim \text{Uniforme}(0,1)
$$

Ici :
- $\Phi(z) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{z}{\sqrt{2}}\right)\right]$ est la fonction de répartition de la loi normale centrée réduite;
- $\Phi^{-1}$ est son inverse (la fonction quantile);
- $\text{erf}(z) = \frac{2}{\sqrt{\pi}}\int_0^z e^{-t^2}dt$ est la fonction d'erreur.

Les échantillons obtenus suivent une loi normale restreinte à $[a,b]$, dont la densité est correctement normalisée sur cet intervalle.

##### La loi de Kumaraswamy
Lorsqu'il faut des échantillons dans l'intervalle unité [0,1], la loi bêta pourrait sembler un choix naturel. Toutefois, l'inverse de sa fonction de répartition n'a pas de forme close. On peut plutôt utiliser la loi de Kumaraswamy comme approximation commode, qui admet une reparamétrisation simple :

$$
x = (1-(1-u^{\alpha})^{1/\beta}), \quad u \sim \text{Uniforme}(0,1)
$$

où :
- $\alpha, \beta > 0$ sont des paramètres de forme qui règlent la distribution;
- $\alpha$ détermine la concentration autour de 0;
- $\beta$ détermine la concentration autour de 1;
- la distribution ressemble à Bêta(α,β), mais sa fonction de répartition et l'inverse de celle-ci se calculent analytiquement.

La loi de Kumaraswamy a pour densité :

$$
f(x; \alpha, \beta) = \alpha\beta x^{\alpha-1}(1-x^{\alpha})^{\beta-1}, \quad x \in [0,1]
$$

##### La loi Gumbel-Softmax

Pour échantillonner une loi catégorielle de probabilités $\{\pi_i\}$, une approche combine un bruit $\text{Gumbel}(0,1)$ avec l'argmax des log-probabilités perturbées :

$$
\text{argmax}_i(\log \pi_i + g_i), \quad g_i \sim \text{Gumbel}(0,1)
$$

Cette approche, connue en apprentissage automatique sous le nom d'astuce Gumbel-Max, repose sur l'échantillonnage d'un bruit de Gumbel à partir de variables aléatoires uniformes par la transformation $g_i = -\log(-\log(u_i))$, où $u_i \sim \text{Uniforme}(0,1)$. Pour voir pourquoi on obtient ainsi des échantillons de la loi catégorielle, considérons la probabilité de choisir la catégorie $i$ :

$$
\begin{align*}
P(\text{argmax}_j(\log \pi_j + g_j) = i) &= P(\log \pi_i + g_i > \log \pi_j + g_j \text{ pour tout } j \neq i) \\
&= P(g_i - g_j > \log \pi_j - \log \pi_i \text{ pour tout } j \neq i)
\end{align*}
$$

Comme la différence de deux variables aléatoires de Gumbel suit une loi logistique, $g_i - g_j \sim \text{Logistique}(0,1)$, et que ces différences sont indépendantes pour des $j$ différents (en raison de l'indépendance des variables de Gumbel d'origine), on peut écrire :

$$
\begin{align*}
P(\text{argmax}_j(\log \pi_j + g_j) = i) &= \prod_{j \neq i} P(g_i - g_j > \log \pi_j - \log \pi_i) \\
&= \prod_{j \neq i} \frac{\pi_i}{\pi_i + \pi_j} = \pi_i
\end{align*}
$$

La dernière égalité demande un peu plus d'algèbre, mais elle découle du fait que ces probabilités doivent avoir une somme de 1 sur tous les $i$.

Nous avons montré que l'astuce Gumbel-Max donne des échantillons exacts d'une loi catégorielle, mais l'opération argmax n'est pas dérivable. Pour des problèmes d'optimisation stochastique de la forme :

$$
\mathbb{E}_{x \sim p(x;\theta)}[f(x)] = \mathbb{E}_{\epsilon \sim \text{Gumbel}(0,1)}[f(g(\epsilon,\theta))]
$$

il faut que $g$ soit dérivable par rapport à $\theta$. Cela conduit à une relaxation continue, où l'on remplace l'argmax strict par une fonction softmax réglée par une température :

$$
z_i = \frac{\exp((\log \pi_i + g_i)/\tau)}{\sum_j \exp((\log \pi_j + g_j)/\tau)}
$$

Lorsque $\tau \to 0$, cette approximation tend vers l'argmax :

$$
\lim_{\tau \to 0} \frac{\exp(x_i/\tau)}{\sum_j \exp(x_j/\tau)} = \begin{cases} 1 & \text{si } x_i = \max_j x_j \\ 0 & \text{sinon} \end{cases}
$$

La distribution obtenue sur le simplexe des probabilités s'appelle la loi Gumbel-Softmax (ou loi Concrete). Le paramètre de température $\tau$ règle le caractère discret des échantillons : de petites valeurs donnent des échantillons plus proches de vecteurs indicateurs (à un seul 1), mais des gradients moins stables, tandis que de grandes valeurs donnent des gradients plus lisses, mais des échantillons plus diffus.


### Analyse numérique des estimateurs de gradient

Examinons le comportement de nos trois estimateurs de gradient pour l'objectif d'optimisation stochastique :

$$J(\theta) = \mathbb{E}_{x \sim \mathcal{N}(\theta,1)}[x^2\theta]$$ 

Pour obtenir une expression analytique de la dérivée, remarquons d'abord qu'on peut factoriser $\theta$ pour obtenir $J(\theta) = \theta\mathbb{E}[x^2]$, où $x \sim \mathcal{N}(\theta,1)$. Par définition de la variance, $\text{Var}(x) = \mathbb{E}[x^2] - (\mathbb{E}[x])^2$, ce qu'on peut réarranger en $\mathbb{E}[x^2] = \text{Var}(x) + (\mathbb{E}[x])^2$. Comme $x \sim \mathcal{N}(\theta,1)$, on a $\text{Var}(x) = 1$ et $\mathbb{E}[x] = \theta$, donc $\mathbb{E}[x^2] = 1 + \theta^2$. On obtient :

$$J(\theta) = \theta(1 + \theta^2)$$

En dérivant maintenant par rapport à $\theta$ à l'aide de la règle du produit, on obtient :

$$\frac{d}{d\theta}J(\theta) = 1 + 3\theta^2$$ 

Pour fixer les idées, on pose $\theta = 1.0$ et on analyse des échantillons tirés par estimation de Monte-Carlo avec des lots de taille 1000 et 1000 essais indépendants. L'évaluation en $\theta = 1$ donne $\frac{d}{d\theta}J(\theta)\big|_{\theta=1} = 1 + 3(1)^2 = 4$, la valeur exacte à laquelle nous comparons nos estimateurs :

1.  On considère d'abord l'estimateur naïf, qui dérive incorrectement l'approximation de Monte-Carlo :

    $$\hat{g}_{\text{naive}}(\theta) = \frac{1}{N}\sum_{i=1}^N x_i^2$$

    Pour $x \sim \mathcal{N}(1,1)$, on a $\mathbb{E}[x^2] = \theta^2 + 1 = 2.0$ et $\mathbb{E}[\hat{g}_{\text{naive}}] = 2.0$. On doit donc s'attendre à un biais d'environ $-2$ dans notre expérience.

2. On calcule ensuite l'estimateur par fonction de score :

    $$\hat{g}_{\text{SF}}(\theta) = \frac{1}{N}\sum_{i=1}^N \left[x_i^2\theta(x_i - \theta) + x_i^2\right]$$

    Cet estimateur est sans biais : $\mathbb{E}[\hat{g}_{\text{SF}}] = 4$.

3. Enfin, au moyen de la reparamétrisation $x = \theta + \epsilon$, où $\epsilon \sim \mathcal{N}(0,1)$, on obtient :

    $$\hat{g}_{\text{RT}}(\theta) = \frac{1}{N}\sum_{i=1}^N \left[2\theta(\theta + \epsilon_i) + (\theta + \epsilon_i)^2\right]$$

    Cet estimateur est lui aussi sans biais : $\mathbb{E}[\hat{g}_{\text{RT}}] = 4$.


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

Les expériences numériques corroborent la théorie. L'estimateur naïf sous-estime systématiquement le vrai gradient de 2.0, bien qu'il conserve une variance relativement faible. Ce biais systématique le rendrait impropre à l'optimisation malgré sa faible variance. L'estimateur par fonction de score corrige ce biais, mais introduit une variance considérable. Bien qu'il soit sans biais, cet estimateur exigerait de nombreux échantillons pour obtenir des estimations fiables du gradient. Enfin, l'astuce de reparamétrisation atteint une variance beaucoup plus faible tout en restant sans biais. Cette expérience n'a qu'une visée didactique, mais elle reproduit ce qu'on observe couramment en pratique : lorsqu'il est applicable, l'estimateur par reparamétrisation tend à donner de meilleurs résultats que son pendant par fonction de score.

## Synthèse et perspectives

L'identité de la fonction de score dérive le logarithme d'une densité et
s'applique même lorsque la variable échantillonnée est discrète, mais sa
variance peut être grande. La reparamétrisation dérive le résultat échantillonné
par rapport à ses paramètres et a généralement une variance plus faible, mais
elle exige un chemin d'échantillonnage dérivable. La comparaison numérique
sépare directement ces propriétés de variance et de biais.

La commande continue régularisée par l'entropie exige à la fois des actions
échantillonnées et un signal de valeur appris. Ces estimateurs peuvent-ils
entraîner un acteur stochastique tout en évitant une intégrale intraitable sur
les actions? [L'apprentissage de politiques régularisé et fondé sur les
résidus](regularized-policy-learning.md) développe SAC, la cohérence des chemins
et des constructions apparentées.

## Autoévaluation

:::{exercise} Limite de la reparamétrisation
:label: ex-pg-check-3

Donnez un cadre où la méthode par fonction de score s'applique, mais où un gradient de reparamétrisation par chemin n'est pas directement disponible.
:::

:::{solution} ex-pg-check-3
:class: dropdown

Une politique à actions discrètes en est l'exemple type : ses échantillons ne sont pas des fonctions dérivables d'un bruit continu, tandis que leurs log-probabilités restent dérivables par rapport aux paramètres de la politique.
:::
